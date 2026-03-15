// styleblock.cu — cuBLASLt HGEMM with optional BIAS epilogue.
//
// Computes C[M,N] = A[M,K] * B[K,N]  (+ bias[M] when BIAS epilogue active)
// for InSwapper-128 style-block convolutions.
//
// Row-major / col-major transposition recipe:
//   C_row[M,N] = A_row[M,K] * B_row[K,N]
//   In col-major: C_col[N,M] = B_col[N,K] * A_col[K,M]
//   A row-major tensor X[r,c] is stored identically to a col-major [c,r]:
//     A_row[M,K] → col-major [K,M], ldb=K  (passed as cuBLASLt "B")
//     B_row[K,N] → col-major [N,K], lda=N  (passed as cuBLASLt "A")
//     C_row[M,N] → col-major [N,M], ldc=N
//   BIAS epilogue adds bias[j] to column j of C_col = row j of C_row ✓
//
// Build (from project root with MSVC env active):
//   python custom_kernels/build_kernels.py
// Or manually:
//   run_with_msvc.bat custom_kernels/inswapper_128/_styleblock_build/build.py

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <mutex>
#include <unordered_map>
#include <stdexcept>

// ── Module-level handle ───────────────────────────────────────────────────────
static cublasLtHandle_t g_handle = nullptr;
static std::mutex       g_init_mx;

static cublasLtHandle_t get_handle() {
    std::lock_guard<std::mutex> lk(g_init_mx);
    if (!g_handle) {
        if (cublasLtCreate(&g_handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("[StyleBlock] cublasLtCreate failed");
    }
    return g_handle;
}

// ── Per-op state ──────────────────────────────────────────────────────────────
struct GemmOp {
    cublasLtMatmulDesc_t   desc    = nullptr;
    cublasLtMatrixLayout_t lA      = nullptr;
    cublasLtMatrixLayout_t lB      = nullptr;
    cublasLtMatrixLayout_t lC      = nullptr;
    cublasLtMatmulAlgo_t   algo    {};
    bool                   has_algo = false;
    void*                  ws       = nullptr;
    size_t                 ws_sz    = 0;
    int64_t                M = 0, K = 0, N = 0;

    GemmOp() = default;
    GemmOp(const GemmOp&) = delete;
    GemmOp& operator=(const GemmOp&) = delete;
    GemmOp(GemmOp&& o) noexcept
        : desc(o.desc), lA(o.lA), lB(o.lB), lC(o.lC),
          algo(o.algo), has_algo(o.has_algo),
          ws(o.ws), ws_sz(o.ws_sz), M(o.M), K(o.K), N(o.N)
    { o.desc=nullptr; o.lA=nullptr; o.lB=nullptr; o.lC=nullptr; o.ws=nullptr; }

    ~GemmOp() {
        if (desc) cublasLtMatmulDescDestroy(desc);
        if (lA)   cublasLtMatrixLayoutDestroy(lA);
        if (lB)   cublasLtMatrixLayoutDestroy(lB);
        if (lC)   cublasLtMatrixLayoutDestroy(lC);
        if (ws)   cudaFree(ws);
    }
};

static std::unordered_map<int64_t, GemmOp> g_ops;
static std::mutex  g_reg_mx;
static int64_t     g_next_id = 1;

// ── Internal builder ──────────────────────────────────────────────────────────
static int64_t _create(int64_t M, int64_t K, int64_t N,
                        const void* bias_ptr /* nullable */)
{
    cublasLtHandle_t h = get_handle();
    GemmOp op;
    op.M = M; op.K = K; op.N = N;
    op.ws_sz = 4 * 1024 * 1024;
    if (cudaMalloc(&op.ws, op.ws_sz) != cudaSuccess)
        throw std::runtime_error("[StyleBlock] workspace cudaMalloc failed");

    cublasLtMatmulDescCreate(&op.desc, CUBLAS_COMPUTE_16F, CUDA_R_16F);

    if (bias_ptr) {
        cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
        cublasLtMatmulDescSetAttribute(op.desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                       &ep, sizeof(ep));
        cudaDataType_t bdt = CUDA_R_16F;
        cublasLtMatmulDescSetAttribute(op.desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                       &bdt, sizeof(bdt));
        cublasLtMatmulDescSetAttribute(op.desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                       &bias_ptr, sizeof(bias_ptr));
    }

    // Layouts (col-major view of row-major PyTorch tensors):
    //   lA: B_row[K,N] → [N,K] col-major, lda=N
    //   lB: A_row[M,K] → [K,M] col-major, ldb=K
    //   lC: C_row[M,N] → [N,M] col-major, ldc=N
    cublasLtMatrixLayoutCreate(&op.lA, CUDA_R_16F, N, K, N);
    cublasLtMatrixLayoutCreate(&op.lB, CUDA_R_16F, K, M, K);
    cublasLtMatrixLayoutCreate(&op.lC, CUDA_R_16F, N, M, N);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &op.ws_sz, sizeof(op.ws_sz));
    cublasLtMatmulHeuristicResult_t res[4];
    int n_res = 0;
    cublasLtMatmulAlgoGetHeuristic(h, op.desc,
                                    op.lA, op.lB, op.lC, op.lC,
                                    pref, 4, res, &n_res);
    cublasLtMatmulPreferenceDestroy(pref);
    if (n_res > 0) { op.algo = res[0].algo; op.has_algo = true; }

    std::lock_guard<std::mutex> lk(g_reg_mx);
    int64_t id = g_next_id++;
    g_ops.emplace(id, std::move(op));
    return id;
}

// ── Public API ────────────────────────────────────────────────────────────────
int64_t create_hgemm_op(int64_t M, int64_t K, int64_t N) {
    return _create(M, K, N, nullptr);
}

int64_t create_hgemm_bias_op(int64_t M, int64_t K, int64_t N,
                               torch::Tensor bias) {
    TORCH_CHECK(bias.is_cuda() && bias.dtype() == torch::kHalf,
                "bias must be CUDA float16");
    TORCH_CHECK(bias.numel() == M, "bias.numel() must equal M");
    return _create(M, K, N, bias.data_ptr());
}

// Run: C[M,N] = A[M,K] * B[K,N]  (bias fused if op was created with _bias variant)
// CUDA-graph safe — only cublasLtMatmul is captured, no descriptor manipulation.
torch::Tensor run_hgemm_op(int64_t op_id,
                             torch::Tensor A,   // [M, K] FP16
                             torch::Tensor B)   // [K, N] FP16
{
    auto& op = g_ops.at(op_id);
    TORCH_CHECK(A.is_cuda() && A.dtype() == torch::kHalf);
    TORCH_CHECK(B.is_cuda() && B.dtype() == torch::kHalf);

    auto C = torch::empty({op.M, op.N}, A.options());

    const __half alpha = __float2half(1.0f);
    const __half beta  = __float2half(0.0f);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    cublasLtMatmul(get_handle(), op.desc,
                   &alpha,
                   B.data_ptr(), op.lA,
                   A.data_ptr(), op.lB,
                   &beta,
                   C.data_ptr(), op.lC,
                   C.data_ptr(), op.lC,
                   op.has_algo ? &op.algo : nullptr,
                   op.ws, op.ws_sz,
                   stream);
    return C;
}

void destroy_hgemm_op(int64_t op_id) {
    std::lock_guard<std::mutex> lk(g_reg_mx);
    g_ops.erase(op_id);
}
