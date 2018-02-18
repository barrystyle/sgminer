/**
 * Proper ethash OpenCL kernel compatible with AMD and NVIDIA
 *
 * (c) tpruvot @ October 2016
 */

#define SPH_COMPACT_BLAKE_64 0

#define ACCESSES   64
#define MAX_OUTPUTS 255u
#define barrier(x) mem_fence(x)


#define WORD_BYTES 4
#define DATASET_BYTES_INIT 536870912
#define DATASET_BYTES_GROWTH 12582912
#define CACHE_BYTES_INIT 8388608
#define CACHE_BYTES_GROWTH 196608
#define EPOCH_LENGTH 400
#define CACHE_MULTIPLIER 64
#define MIX_BYTES 64
#define HASH_BYTES 32
#define DATASET_PARENTS 256
#define CACHE_ROUNDS 3
#define ACCESSES 64
#define FNV_PRIME 0x01000193U

#define fnv(x, y) ((x) * FNV_PRIME ^ (y)) % (0xffffffff)
#define fnv_reduce(v) fnv(fnv(fnv(v.x, v.y), v.z), v.w)

#ifdef cl_nv_pragma_unroll
#define NVIDIA
#else
#pragma OPENCL EXTENSION cl_amd_media_ops2 : enable
#define ROTL64_1(x, y)  amd_bitalign((x), (x).s10, (32U - y))
#define ROTL64_2(x, y)  amd_bitalign((x).s10, (x), (32U - y))
#define ROTL64_8(x, y)  amd_bitalign((x), (x).s10, 24U)
#define BFE(x, start, len)  amd_bfe(x, start, len)
#endif

#ifdef NVIDIA
static inline uint2 rol2(const uint2 a, const uint offset) {
	uint2 r;
	asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r.x) : "r"(a.y), "r"(a.x), "r"(offset));
	asm("shf.l.wrap.b32 %0, %1, %2, %3;" : "=r"(r.y) : "r"(a.x), "r"(a.y), "r"(offset));
	return r;
}
static inline uint2 ror2(const uint2 a, const uint offset) {
	uint2 r;
	asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r.x) : "r"(a.x), "r"(a.y), "r"(offset));
	asm("shf.r.wrap.b32 %0, %1, %2, %3;" : "=r"(r.y) : "r"(a.y), "r"(a.x), "r"(offset));
	return r;
}
static inline uint2 rol8(const uint2 a) {
	uint2 r;
	asm("prmt.b32 %0, %1, %2, 0x6543;" : "=r"(r.x) : "r"(a.y), "r"(a.x));
	asm("prmt.b32 %0, %1, %2, 0x2107;" : "=r"(r.y) : "r"(a.y), "r"(a.x));
	return r;
}

#define ROTL64_1(x, y) rol2(x, y)
#define ROTL64_2(x, y) ror2(x, (32U - y))
#define ROTL64_8(x, y) rol8(x)

static inline uint nv_bfe(const uint a, const uint start, const uint len) {
	uint r;
	asm("bfe.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(start), "r"(len));
	return r;
}
#define BFE(x, start, len) nv_bfe(x, start, len)
#endif /* NVIDIA */

// BEGIN BLAKE256

// Blake256 Macros
__constant static const uint sigma[16][16] = {
	{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
	{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
	{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
	{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
	{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 },
	{ 12,  5,  1, 15, 14, 13,  4, 10,  0,  7,  6,  3,  9,  2,  8, 11 },
	{ 13, 11,  7, 14, 12,  1,  3,  9,  5,  0, 15,  4,  8,  6,  2, 10 },
	{  6, 15, 14,  9, 11,  3,  0,  8, 12,  2, 13,  7,  1,  4, 10,  5 },
	{ 10,  2,  8,  4,  7,  6,  1,  5, 15, 11,  9, 14,  3, 12, 13,  0 },
	{  0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15 },
	{ 14, 10,  4,  8,  9, 15, 13,  6,  1, 12,  0,  2, 11,  7,  5,  3 },
	{ 11,  8, 12,  0,  5,  2, 15, 13, 10, 14,  3,  6,  7,  1,  9,  4 },
	{  7,  9,  3,  1, 13, 12, 11, 14,  2,  6,  5, 10,  4,  0, 15,  8 },
	{  9,  0,  5,  7,  2,  4, 10, 15, 14,  1, 11, 12,  6,  8,  3, 13 },
	{  2, 12,  6, 10,  0, 11,  8,  3,  4, 13,  7,  5, 15, 14,  1,  9 }
};

__constant static const uint  c_u256[16] = {
	0x243F6A88, 0x85A308D3,
	0x13198A2E, 0x03707344,
	0xA4093822, 0x299F31D0,
	0x082EFA98, 0xEC4E6C89,
	0x452821E6, 0x38D01377,
	0xBE5466CF, 0x34E90C6C,
	0xC0AC29B7, 0xC97C50DD,
	0x3F84D5B5, 0xB5470917
};

#define SPH_C32(x)    ((uint)(x))
#define SPH_T32(x)    ((x) & SPH_C32(0xFFFFFFFF))

#define SPH_ROTL32(x, n)   SPH_T32(((x) << (n)) | ((x) >> (32 - (n))))
#define SPH_ROTR32(x, n)   SPH_ROTL32(x, (32 - (n)))

inline uint sph_bswap32(uint in_swap)
{
   return (uint)(((in_swap >> 24) & 0x000000ff) |
              ((in_swap >>  8) & 0x0000ff00) |
              ((in_swap <<  8) & 0x00ff0000) |
              ((in_swap << 24) & 0xff000000));
}

#define BLAKE256_GS(m0, m1, c0, c1, a, b, c, d)   do { \
		a = SPH_T32(a + b + (m0 ^ c1)); \
		d = SPH_ROTR32(d ^ a, 16); \
		c = SPH_T32(c + d); \
		b = SPH_ROTR32(b ^ c, 12); \
		a = SPH_T32(a + b + (m1 ^ c0)); \
		d = SPH_ROTR32(d ^ a, 8); \
		c = SPH_T32(c + d); \
		b = SPH_ROTR32(b ^ c, 7); \
	} while (0)

#define BLAKE256_STATE \
uint H0, H1, H2, H3, H4, H5, H6, H7, S0, S1, S2, S3, T0, T1;
#define INIT_BLAKE256_STATE \
H0 = SPH_C32(0x6a09e667); \
H1 = SPH_C32(0xbb67ae85); \
H2 = SPH_C32(0x3c6ef372); \
H3 = SPH_C32(0xa54ff53a); \
H4 = SPH_C32(0x510e527f); \
H5 = SPH_C32(0x9b05688c); \
H6 = SPH_C32(0x1f83d9ab); \
H7 = SPH_C32(0x5be0cd19); \
S0 = 0; \
S1 = 0; \
S2 = 0; \
S3 = 0; \
T0 = 0; \
T1 = 0;

#define BLAKE32_ROUNDS 14

#define BLAKE256_COMPRESS32_STATE \
uint M[16]; \
uint V[16]; \
uint R;

#define BLAKE256_COMPRESS32(b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15) \
V[0] = H0; \
V[1] = H1; \
V[2] = H2; \
V[3] = H3; \
V[4] = H4; \
V[5] = H5; \
V[6] = H6; \
V[7] = H7; \
V[8] = S0 ^ c_u256[0]; \
V[9] = S1 ^ c_u256[1]; \
V[10] = S2 ^ c_u256[2]; \
V[11] = S3 ^ c_u256[3]; \
V[12] = T0 ^ c_u256[4]; \
V[13] = T0 ^ c_u256[5]; \
V[14] = T1 ^ c_u256[6]; \
V[15] = T1 ^ c_u256[7]; \
M[0x0] = b0; \
M[0x1] = b1; \
M[0x2] = b2; \
M[0x3] = b3; \
M[0x4] = b4; \
M[0x5] = b5; \
M[0x6] = b6; \
M[0x7] = b7; \
M[0x8] = b8; \
M[0x9] = b9; \
M[0xA] = b10; \
M[0xB] = b11; \
M[0xC] = b12; \
M[0xD] = b13; \
M[0xE] = b14; \
M[0xF] = b15; \
for (R=0; R< BLAKE32_ROUNDS; R++) { \
	BLAKE256_GS(M[sigma[R][0x0]], M[sigma[R][0x1]], c_u256[sigma[R][0x0]], c_u256[sigma[R][0x1]], V[0x0], V[0x4], V[0x8], V[0xC]); \
	BLAKE256_GS(M[sigma[R][0x2]], M[sigma[R][0x3]], c_u256[sigma[R][0x2]], c_u256[sigma[R][0x3]], V[0x1], V[0x5], V[0x9], V[0xD]); \
	BLAKE256_GS(M[sigma[R][0x4]], M[sigma[R][0x5]], c_u256[sigma[R][0x4]], c_u256[sigma[R][0x5]], V[0x2], V[0x6], V[0xA], V[0xE]); \
	BLAKE256_GS(M[sigma[R][0x6]], M[sigma[R][0x7]], c_u256[sigma[R][0x6]], c_u256[sigma[R][0x7]], V[0x3], V[0x7], V[0xB], V[0xF]); \
	BLAKE256_GS(M[sigma[R][0x8]], M[sigma[R][0x9]], c_u256[sigma[R][0x8]], c_u256[sigma[R][0x9]], V[0x0], V[0x5], V[0xA], V[0xF]); \
	BLAKE256_GS(M[sigma[R][0xA]], M[sigma[R][0xB]], c_u256[sigma[R][0xA]], c_u256[sigma[R][0xB]], V[0x1], V[0x6], V[0xB], V[0xC]); \
	BLAKE256_GS(M[sigma[R][0xC]], M[sigma[R][0xD]], c_u256[sigma[R][0xC]], c_u256[sigma[R][0xD]], V[0x2], V[0x7], V[0x8], V[0xD]); \
	BLAKE256_GS(M[sigma[R][0xE]], M[sigma[R][0xF]], c_u256[sigma[R][0xE]], c_u256[sigma[R][0xF]], V[0x3], V[0x4], V[0x9], V[0xE]); \
} \
H0 ^= S0 ^ V[0] ^ V[8]; \
H1 ^= S1 ^ V[1] ^ V[9]; \
H2 ^= S2 ^ V[2] ^ V[10]; \
H3 ^= S3 ^ V[3] ^ V[11]; \
H4 ^= S0 ^ V[4] ^ V[12]; \
H5 ^= S1 ^ V[5] ^ V[13]; \
H6 ^= S2 ^ V[6] ^ V[14]; \
H7 ^= S3 ^ V[7] ^ V[15];

// END BLAKE256

//#define DEBUG

// Cache node
typedef union _Node
{
	uint dwords[8];
	uint2 qwords[4];
	uint4 dqwords[2];
} Node; // NOTE: should be HASH_BYTES long


typedef union {
	uint  uints[32 / sizeof(uint)];
	ulong ulongs[32 / sizeof(ulong)];
} hash32_t;

typedef union {
	uint   uints[128 / sizeof(uint)];
	ulong  ulongs[128 / sizeof(ulong)];
	uint2  uint2s[128 / sizeof(uint2)];
	uint4  uint4s[128 / sizeof(uint4)];
	uint8  uint8s[128 / sizeof(uint8)];
	uint16 uint16s[128 / sizeof(uint16)];
	ulong8 ulong8s[128 / sizeof(ulong8)];
} hash128_t;

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search(
	__global volatile uint* restrict g_output,
	__constant hash32_t const* g_header,
	__global hash128_t const* g_dag,
	uint DAG_SIZE,
	ulong target,
	uint isolate
	)
{
	const uint gid = get_global_id(0); // i.e. nonce
	const uint thread_id = get_local_id(0) & 3U;
	const uint hash_id = get_local_id(0) >> 2U;

	// TODO: implement
}

__kernel void GenerateDAG(uint start, __global const uint16 *_Cache, __global uint16 *_DAG, uint LIGHT_SIZE)
{
	__global const Node *Cache = (__global const Node *) _Cache;
	__global Node *DAG = (__global Node *) _DAG;
	uint NodeIdx = start + get_global_id(0);

	Node DAGNode = Cache[NodeIdx % LIGHT_SIZE];
	DAGNode.dwords[0] ^= NodeIdx;

	BLAKE256_STATE;
	BLAKE256_COMPRESS32_STATE;

	//printf("generateDAG %u\n", NodeIdx);

	// Apply blake to DAGNode

	INIT_BLAKE256_STATE;
	// Blake hash full input
	// blake close - t0==0 case
	T0 = SPH_C32(0xFFFFFE00) + 256;
	T1 = SPH_C32(0xFFFFFFFF);
	T0 = SPH_T32(T0 + 512);
	T1 = SPH_T32(T1 + 1);

	BLAKE256_COMPRESS32(sph_bswap32(DAGNode.dwords[0]),sph_bswap32(DAGNode.dwords[1]),sph_bswap32(DAGNode.dwords[2]),sph_bswap32(DAGNode.dwords[3]),sph_bswap32(DAGNode.dwords[4]),sph_bswap32(DAGNode.dwords[5]),sph_bswap32(DAGNode.dwords[6]),sph_bswap32(DAGNode.dwords[7]),2147483648,0,0,0,0,1,0,256);
	DAGNode.dwords[0] = sph_bswap32(H0);
	DAGNode.dwords[1] = sph_bswap32(H1);
	DAGNode.dwords[2] = sph_bswap32(H2);
	DAGNode.dwords[3] = sph_bswap32(H3);
	DAGNode.dwords[4] = sph_bswap32(H4);
	DAGNode.dwords[5] = sph_bswap32(H5);
	DAGNode.dwords[6] = sph_bswap32(H6);
	DAGNode.dwords[7] = sph_bswap32(H7);

	for (uint parent = 0; parent < DATASET_PARENTS; ++parent)
	{
		// Calculate parent
		uint ParentIdx = fnv(NodeIdx ^ parent, DAGNode.dwords[parent & 7]) % LIGHT_SIZE; // NOTE: LIGHT_SIZE == items, &7 == %8
		__global const Node *ParentNode = Cache + ParentIdx;

		#pragma unroll
		for (uint x = 0; x < 2; ++x)
		{
			// NOTE: fnv, we're basically operating on 4 ints at a time here
			DAGNode.dqwords[x] *= (uint4)(FNV_PRIME);
			DAGNode.dqwords[x] ^= ParentNode->dwords[0];
			DAGNode.dqwords[x] %= SPH_C32(0xffffffff);
		}
	}
	
	// Apply final blake to NodeIdx

	INIT_BLAKE256_STATE;
	// Blake hash full input
	// blake close - t0==0 case
	T0 = SPH_C32(0xFFFFFE00) + 256;
	T1 = SPH_C32(0xFFFFFFFF);
	T0 = SPH_T32(T0 + 512);
	T1 = SPH_T32(T1 + 1);

	BLAKE256_COMPRESS32(sph_bswap32(DAGNode.dwords[0]),sph_bswap32(DAGNode.dwords[1]),sph_bswap32(DAGNode.dwords[2]),sph_bswap32(DAGNode.dwords[3]),sph_bswap32(DAGNode.dwords[4]),sph_bswap32(DAGNode.dwords[5]),sph_bswap32(DAGNode.dwords[6]),sph_bswap32(DAGNode.dwords[7]),2147483648,0,0,0,0,1,0,256);
	DAGNode.dwords[0] = sph_bswap32(H0);
	DAGNode.dwords[1] = sph_bswap32(H1);
	DAGNode.dwords[2] = sph_bswap32(H2);
	DAGNode.dwords[3] = sph_bswap32(H3);
	DAGNode.dwords[4] = sph_bswap32(H4);
	DAGNode.dwords[5] = sph_bswap32(H5);
	DAGNode.dwords[6] = sph_bswap32(H6);
	DAGNode.dwords[7] = sph_bswap32(H7);


	DAG[NodeIdx] = DAGNode;
}
