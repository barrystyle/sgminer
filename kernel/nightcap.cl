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

#define MAX_NONCE_OUTPUTS 255
#define MAX_HASH_OUTPUTS  256

// DAG Cache node
typedef union _Node
{
	uint dwords[8];
	uint4 dqwords[2];
} Node; // NOTE: should be HASH_BYTES long


// Output hash
typedef union {
	unsigned char h1[32];
	uint h4[8];
	ulong h8[4];
} hash32_t;

//#define fnv(x, y) ((x) * FNV_PRIME ^ (y)) % (0xffffffff)
//#define fnv_reduce(v) fnv(fnv(fnv(v.x, v.y), v.z), v.w)

uint fnv(const uint v1, const uint v2) {
	return ((v1 * FNV_PRIME) ^ v2) % (0xffffffff);
}

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

//
// BEGIN BLAKE256
//

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

#define SPH_C64(x)    ((ulong)(x ## UL))
#define SPH_T64(x)    ((x) & SPH_C64(0xFFFFFFFFFFFFFFFF))

#define SPH_ROTL64(x, n)   SPH_T64(((x) << (n)) | ((x) >> (64 - (n))))
#define SPH_ROTR64(x, n)   SPH_ROTL64(x, (64 - (n)))

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
V[8] = S0 ^ 0x243F6A88; \
V[9] = S1 ^ 0x85A308D3; \
V[10] = S2 ^ 0x13198A2E; \
V[11] = S3 ^ 0x3707344; \
V[12] = T0 ^ 0xA4093822; \
V[13] = T0 ^ 0x299F31D0; \
V[14] = T1 ^ 0x82EFA98; \
V[15] = T1 ^ 0xEC4E6C89; \
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
BLAKE256_GS(M[0], M[1], 0x243F6A88, 0x85A308D3, V[0x0], V[0x4], V[0x8], V[0xC]); \
BLAKE256_GS(M[2], M[3], 0x13198A2E, 0x3707344, V[0x1], V[0x5], V[0x9], V[0xD]); \
BLAKE256_GS(M[4], M[5], 0xA4093822, 0x299F31D0, V[0x2], V[0x6], V[0xA], V[0xE]); \
BLAKE256_GS(M[6], M[7], 0x82EFA98, 0xEC4E6C89, V[0x3], V[0x7], V[0xB], V[0xF]); \
BLAKE256_GS(M[8], M[9], 0x452821E6, 0x38D01377, V[0x0], V[0x5], V[0xA], V[0xF]); \
BLAKE256_GS(M[10], M[11], 0xBE5466CF, 0x34E90C6C, V[0x1], V[0x6], V[0xB], V[0xC]); \
BLAKE256_GS(M[12], M[13], 0xC0AC29B7, 0xC97C50DD, V[0x2], V[0x7], V[0x8], V[0xD]); \
BLAKE256_GS(M[14], M[15], 0x3F84D5B5, 0xB5470917, V[0x3], V[0x4], V[0x9], V[0xE]); \
BLAKE256_GS(M[14], M[10], 0x3F84D5B5, 0xBE5466CF, V[0x0], V[0x4], V[0x8], V[0xC]); \
BLAKE256_GS(M[4], M[8], 0xA4093822, 0x452821E6, V[0x1], V[0x5], V[0x9], V[0xD]); \
BLAKE256_GS(M[9], M[15], 0x38D01377, 0xB5470917, V[0x2], V[0x6], V[0xA], V[0xE]); \
BLAKE256_GS(M[13], M[6], 0xC97C50DD, 0x82EFA98, V[0x3], V[0x7], V[0xB], V[0xF]); \
BLAKE256_GS(M[1], M[12], 0x85A308D3, 0xC0AC29B7, V[0x0], V[0x5], V[0xA], V[0xF]); \
BLAKE256_GS(M[0], M[2], 0x243F6A88, 0x13198A2E, V[0x1], V[0x6], V[0xB], V[0xC]); \
BLAKE256_GS(M[11], M[7], 0x34E90C6C, 0xEC4E6C89, V[0x2], V[0x7], V[0x8], V[0xD]); \
BLAKE256_GS(M[5], M[3], 0x299F31D0, 0x3707344, V[0x3], V[0x4], V[0x9], V[0xE]); \
BLAKE256_GS(M[11], M[8], 0x34E90C6C, 0x452821E6, V[0x0], V[0x4], V[0x8], V[0xC]); \
BLAKE256_GS(M[12], M[0], 0xC0AC29B7, 0x243F6A88, V[0x1], V[0x5], V[0x9], V[0xD]); \
BLAKE256_GS(M[5], M[2], 0x299F31D0, 0x13198A2E, V[0x2], V[0x6], V[0xA], V[0xE]); \
BLAKE256_GS(M[15], M[13], 0xB5470917, 0xC97C50DD, V[0x3], V[0x7], V[0xB], V[0xF]); \
BLAKE256_GS(M[10], M[14], 0xBE5466CF, 0x3F84D5B5, V[0x0], V[0x5], V[0xA], V[0xF]); \
BLAKE256_GS(M[3], M[6], 0x3707344, 0x82EFA98, V[0x1], V[0x6], V[0xB], V[0xC]); \
BLAKE256_GS(M[7], M[1], 0xEC4E6C89, 0x85A308D3, V[0x2], V[0x7], V[0x8], V[0xD]); \
BLAKE256_GS(M[9], M[4], 0x38D01377, 0xA4093822, V[0x3], V[0x4], V[0x9], V[0xE]); \
BLAKE256_GS(M[7], M[9], 0xEC4E6C89, 0x38D01377, V[0x0], V[0x4], V[0x8], V[0xC]); \
BLAKE256_GS(M[3], M[1], 0x3707344, 0x85A308D3, V[0x1], V[0x5], V[0x9], V[0xD]); \
BLAKE256_GS(M[13], M[12], 0xC97C50DD, 0xC0AC29B7, V[0x2], V[0x6], V[0xA], V[0xE]); \
BLAKE256_GS(M[11], M[14], 0x34E90C6C, 0x3F84D5B5, V[0x3], V[0x7], V[0xB], V[0xF]); \
BLAKE256_GS(M[2], M[6], 0x13198A2E, 0x82EFA98, V[0x0], V[0x5], V[0xA], V[0xF]); \
BLAKE256_GS(M[5], M[10], 0x299F31D0, 0xBE5466CF, V[0x1], V[0x6], V[0xB], V[0xC]); \
BLAKE256_GS(M[4], M[0], 0xA4093822, 0x243F6A88, V[0x2], V[0x7], V[0x8], V[0xD]); \
BLAKE256_GS(M[15], M[8], 0xB5470917, 0x452821E6, V[0x3], V[0x4], V[0x9], V[0xE]); \
BLAKE256_GS(M[9], M[0], 0x38D01377, 0x243F6A88, V[0x0], V[0x4], V[0x8], V[0xC]); \
BLAKE256_GS(M[5], M[7], 0x299F31D0, 0xEC4E6C89, V[0x1], V[0x5], V[0x9], V[0xD]); \
BLAKE256_GS(M[2], M[4], 0x13198A2E, 0xA4093822, V[0x2], V[0x6], V[0xA], V[0xE]); \
BLAKE256_GS(M[10], M[15], 0xBE5466CF, 0xB5470917, V[0x3], V[0x7], V[0xB], V[0xF]); \
BLAKE256_GS(M[14], M[1], 0x3F84D5B5, 0x85A308D3, V[0x0], V[0x5], V[0xA], V[0xF]); \
BLAKE256_GS(M[11], M[12], 0x34E90C6C, 0xC0AC29B7, V[0x1], V[0x6], V[0xB], V[0xC]); \
BLAKE256_GS(M[6], M[8], 0x82EFA98, 0x452821E6, V[0x2], V[0x7], V[0x8], V[0xD]); \
BLAKE256_GS(M[3], M[13], 0x3707344, 0xC97C50DD, V[0x3], V[0x4], V[0x9], V[0xE]); \
BLAKE256_GS(M[2], M[12], 0x13198A2E, 0xC0AC29B7, V[0x0], V[0x4], V[0x8], V[0xC]); \
BLAKE256_GS(M[6], M[10], 0x82EFA98, 0xBE5466CF, V[0x1], V[0x5], V[0x9], V[0xD]); \
BLAKE256_GS(M[0], M[11], 0x243F6A88, 0x34E90C6C, V[0x2], V[0x6], V[0xA], V[0xE]); \
BLAKE256_GS(M[8], M[3], 0x452821E6, 0x3707344, V[0x3], V[0x7], V[0xB], V[0xF]); \
BLAKE256_GS(M[4], M[13], 0xA4093822, 0xC97C50DD, V[0x0], V[0x5], V[0xA], V[0xF]); \
BLAKE256_GS(M[7], M[5], 0xEC4E6C89, 0x299F31D0, V[0x1], V[0x6], V[0xB], V[0xC]); \
BLAKE256_GS(M[15], M[14], 0xB5470917, 0x3F84D5B5, V[0x2], V[0x7], V[0x8], V[0xD]); \
BLAKE256_GS(M[1], M[9], 0x85A308D3, 0x38D01377, V[0x3], V[0x4], V[0x9], V[0xE]); \
BLAKE256_GS(M[12], M[5], 0xC0AC29B7, 0x299F31D0, V[0x0], V[0x4], V[0x8], V[0xC]); \
BLAKE256_GS(M[1], M[15], 0x85A308D3, 0xB5470917, V[0x1], V[0x5], V[0x9], V[0xD]); \
BLAKE256_GS(M[14], M[13], 0x3F84D5B5, 0xC97C50DD, V[0x2], V[0x6], V[0xA], V[0xE]); \
BLAKE256_GS(M[4], M[10], 0xA4093822, 0xBE5466CF, V[0x3], V[0x7], V[0xB], V[0xF]); \
BLAKE256_GS(M[0], M[7], 0x243F6A88, 0xEC4E6C89, V[0x0], V[0x5], V[0xA], V[0xF]); \
BLAKE256_GS(M[6], M[3], 0x82EFA98, 0x3707344, V[0x1], V[0x6], V[0xB], V[0xC]); \
BLAKE256_GS(M[9], M[2], 0x38D01377, 0x13198A2E, V[0x2], V[0x7], V[0x8], V[0xD]); \
BLAKE256_GS(M[8], M[11], 0x452821E6, 0x34E90C6C, V[0x3], V[0x4], V[0x9], V[0xE]); \
BLAKE256_GS(M[13], M[11], 0xC97C50DD, 0x34E90C6C, V[0x0], V[0x4], V[0x8], V[0xC]); \
BLAKE256_GS(M[7], M[14], 0xEC4E6C89, 0x3F84D5B5, V[0x1], V[0x5], V[0x9], V[0xD]); \
BLAKE256_GS(M[12], M[1], 0xC0AC29B7, 0x85A308D3, V[0x2], V[0x6], V[0xA], V[0xE]); \
BLAKE256_GS(M[3], M[9], 0x3707344, 0x38D01377, V[0x3], V[0x7], V[0xB], V[0xF]); \
BLAKE256_GS(M[5], M[0], 0x299F31D0, 0x243F6A88, V[0x0], V[0x5], V[0xA], V[0xF]); \
BLAKE256_GS(M[15], M[4], 0xB5470917, 0xA4093822, V[0x1], V[0x6], V[0xB], V[0xC]); \
BLAKE256_GS(M[8], M[6], 0x452821E6, 0x82EFA98, V[0x2], V[0x7], V[0x8], V[0xD]); \
BLAKE256_GS(M[2], M[10], 0x13198A2E, 0xBE5466CF, V[0x3], V[0x4], V[0x9], V[0xE]); \
H0 ^= S0 ^ V[0] ^ V[8]; \
H1 ^= S1 ^ V[1] ^ V[9]; \
H2 ^= S2 ^ V[2] ^ V[10]; \
H3 ^= S3 ^ V[3] ^ V[11]; \
H4 ^= S0 ^ V[4] ^ V[12]; \
H5 ^= S1 ^ V[5] ^ V[13]; \
H6 ^= S2 ^ V[6] ^ V[14]; \
H7 ^= S3 ^ V[7] ^ V[15];

//
// END BLAKE256
//


//
// BEGIN KECCAK32
//

__constant static const ulong RC[] = {
  SPH_C64(0x0000000000000001), SPH_C64(0x0000000000008082),
  SPH_C64(0x800000000000808A), SPH_C64(0x8000000080008000),
  SPH_C64(0x000000000000808B), SPH_C64(0x0000000080000001),
  SPH_C64(0x8000000080008081), SPH_C64(0x8000000000008009),
  SPH_C64(0x000000000000008A), SPH_C64(0x0000000000000088),
  SPH_C64(0x0000000080008009), SPH_C64(0x000000008000000A),
  SPH_C64(0x000000008000808B), SPH_C64(0x800000000000008B),
  SPH_C64(0x8000000000008089), SPH_C64(0x8000000000008003),
  SPH_C64(0x8000000000008002), SPH_C64(0x8000000000000080),
  SPH_C64(0x000000000000800A), SPH_C64(0x800000008000000A),
  SPH_C64(0x8000000080008081), SPH_C64(0x8000000000008080),
  SPH_C64(0x0000000080000001), SPH_C64(0x8000000080008008)
};


inline void keccak_block(ulong *s) {
	size_t i;
	ulong t[5], u[5], v, w;

	for (i = 0; i < 24; i++) {
		/* theta: c = a[0,i] ^ a[1,i] ^ .. a[4,i] */
		t[0] = s[0] ^ s[5] ^ s[10] ^ s[15] ^ s[20];
		t[1] = s[1] ^ s[6] ^ s[11] ^ s[16] ^ s[21];
		t[2] = s[2] ^ s[7] ^ s[12] ^ s[17] ^ s[22];
		t[3] = s[3] ^ s[8] ^ s[13] ^ s[18] ^ s[23];
		t[4] = s[4] ^ s[9] ^ s[14] ^ s[19] ^ s[24];

		/* theta: d[i] = c[i+4] ^ rotl(c[i+1],1) */
		u[0] = t[4] ^ SPH_ROTL64(t[1], 1);
		u[1] = t[0] ^ SPH_ROTL64(t[2], 1);
		u[2] = t[1] ^ SPH_ROTL64(t[3], 1);
		u[3] = t[2] ^ SPH_ROTL64(t[4], 1);
		u[4] = t[3] ^ SPH_ROTL64(t[0], 1);

		/* theta: a[0,i], a[1,i], .. a[4,i] ^= d[i] */
		s[0] ^= u[0]; s[5] ^= u[0]; s[10] ^= u[0]; s[15] ^= u[0]; s[20] ^= u[0];
		s[1] ^= u[1]; s[6] ^= u[1]; s[11] ^= u[1]; s[16] ^= u[1]; s[21] ^= u[1];
		s[2] ^= u[2]; s[7] ^= u[2]; s[12] ^= u[2]; s[17] ^= u[2]; s[22] ^= u[2];
		s[3] ^= u[3]; s[8] ^= u[3]; s[13] ^= u[3]; s[18] ^= u[3]; s[23] ^= u[3];
		s[4] ^= u[4]; s[9] ^= u[4]; s[14] ^= u[4]; s[19] ^= u[4]; s[24] ^= u[4];

		/* rho pi: b[..] = rotl(a[..], ..) */
		v = s[1];
		s[1] = SPH_ROTL64(s[6], 44);
		s[6] = SPH_ROTL64(s[9], 20);
		s[9] = SPH_ROTL64(s[22], 61);
		s[22] = SPH_ROTL64(s[14], 39);
		s[14] = SPH_ROTL64(s[20], 18);
		s[20] = SPH_ROTL64(s[2], 62);
		s[2] = SPH_ROTL64(s[12], 43);
		s[12] = SPH_ROTL64(s[13], 25);
		s[13] = SPH_ROTL64(s[19], 8);
		s[19] = SPH_ROTL64(s[23], 56);
		s[23] = SPH_ROTL64(s[15], 41);
		s[15] = SPH_ROTL64(s[4], 27);
		s[4] = SPH_ROTL64(s[24], 14);
		s[24] = SPH_ROTL64(s[21], 2);
		s[21] = SPH_ROTL64(s[8], 55);
		s[8] = SPH_ROTL64(s[16], 45);
		s[16] = SPH_ROTL64(s[5], 36);
		s[5] = SPH_ROTL64(s[3], 28);
		s[3] = SPH_ROTL64(s[18], 21);
		s[18] = SPH_ROTL64(s[17], 15);
		s[17] = SPH_ROTL64(s[11], 10);
		s[11] = SPH_ROTL64(s[7], 6);
		s[7] = SPH_ROTL64(s[10], 3);
		s[10] = SPH_ROTL64(v, 1);

		v = s[0]; w = s[1]; s[0] ^= (~w) & s[2]; s[1] ^= (~s[2]) & s[3]; s[2] ^= (~s[3]) & s[4]; s[3] ^= (~s[4]) & v; s[4] ^= (~v) & w;
		v = s[5]; w = s[6]; s[5] ^= (~w) & s[7]; s[6] ^= (~s[7]) & s[8]; s[7] ^= (~s[8]) & s[9]; s[8] ^= (~s[9]) & v; s[9] ^= (~v) & w;
		v = s[10]; w = s[11]; s[10] ^= (~w) & s[12]; s[11] ^= (~s[12]) & s[13]; s[12] ^= (~s[13]) & s[14]; s[13] ^= (~s[14]) & v; s[14] ^= (~v) & w;
		v = s[15]; w = s[16]; s[15] ^= (~w) & s[17]; s[16] ^= (~s[17]) & s[18]; s[17] ^= (~s[18]) & s[19]; s[18] ^= (~s[19]) & v; s[19] ^= (~v) & w;
		v = s[20]; w = s[21]; s[20] ^= (~w) & s[22]; s[21] ^= (~s[22]) & s[23]; s[22] ^= (~s[23]) & s[24]; s[23] ^= (~s[24]) & v; s[24] ^= (~v) & w;

		s[0] ^= RC[i];
	}
};

//
// END KECCAK32
//


//
// BEGIN SKEIN256
//


__constant static const ulong SKEIN_IV512[] = {
  SPH_C64(0x4903ADFF749C51CE), SPH_C64(0x0D95DE399746DF03),
  SPH_C64(0x8FD1934127C79BCE), SPH_C64(0x9A255629FF352CB1),
  SPH_C64(0x5DB62599DF6CA7B0), SPH_C64(0xEABE394CA9D5C3F4),
  SPH_C64(0x991112C71A75B523), SPH_C64(0xAE18A40B660FCC33)
};

__constant static const ulong SKEIN_IV512_256[8] = {
	0xCCD044A12FDB3E13UL, 0xE83590301A79A9EBUL,
	0x55AEA0614F816E6FUL, 0x2A2767A4AE9B94DBUL,
	0xEC06025E74DD7683UL, 0xE7A436CDC4746251UL,
	0xC36FBAF9393AD185UL, 0x3EEDBA1833EDFC13UL
};

__constant static const int ROT256[8][4] = {
	{ 46, 36, 19, 37 },
	{ 33, 27, 14, 42 },
	{ 17, 49, 36, 39 },
	{ 44, 9, 54, 56  },
	{ 39, 30, 34, 24 },
	{ 13, 50, 10, 17 },
	{ 25, 29, 39, 43 },
	{ 8, 35, 56, 22  }
};

__constant static const ulong skein_ks_parity = 0x1BD11BDAA9FC1A22;

__constant static const ulong t12[6] =
{ 0x20UL,
0xf000000000000000UL,
0xf000000000000020UL,
0x08UL,
0xff00000000000000UL,
0xff00000000000008UL
};


#define Round512(p0,p1,p2,p3,p4,p5,p6,p7,ROT)  { \
p0 += p1; p1 = SPH_ROTL64(p1, ROT256[ROT][0]);  p1 ^= p0; \
p2 += p3; p3 = SPH_ROTL64(p3, ROT256[ROT][1]);  p3 ^= p2; \
p4 += p5; p5 = SPH_ROTL64(p5, ROT256[ROT][2]);  p5 ^= p4; \
p6 += p7; p7 = SPH_ROTL64(p7, ROT256[ROT][3]);  p7 ^= p6; \
} 

#define Round_8_512(p0, p1, p2, p3, p4, p5, p6, p7, R) { \
	    Round512(p0, p1, p2, p3, p4, p5, p6, p7, 0); \
	    Round512(p2, p1, p4, p7, p6, p5, p0, p3, 1); \
	    Round512(p4, p1, p6, p3, p0, p5, p2, p7, 2); \
	    Round512(p6, p1, p0, p7, p2, p5, p4, p3, 3); \
	    p0 += h[((R)+0) % 9]; \
      p1 += h[((R)+1) % 9]; \
      p2 += h[((R)+2) % 9]; \
      p3 += h[((R)+3) % 9]; \
      p4 += h[((R)+4) % 9]; \
      p5 += h[((R)+5) % 9] + t[((R)+0) % 3]; \
      p6 += h[((R)+6) % 9] + t[((R)+1) % 3]; \
      p7 += h[((R)+7) % 9] + R; \
		Round512(p0, p1, p2, p3, p4, p5, p6, p7, 4); \
		Round512(p2, p1, p4, p7, p6, p5, p0, p3, 5); \
		Round512(p4, p1, p6, p3, p0, p5, p2, p7, 6); \
		Round512(p6, p1, p0, p7, p2, p5, p4, p3, 7); \
		p0 += h[((R)+1) % 9]; \
		p1 += h[((R)+2) % 9]; \
		p2 += h[((R)+3) % 9]; \
		p3 += h[((R)+4) % 9]; \
		p4 += h[((R)+5) % 9]; \
		p5 += h[((R)+6) % 9] + t[((R)+1) % 3]; \
		p6 += h[((R)+7) % 9] + t[((R)+2) % 3]; \
		p7 += h[((R)+8) % 9] + (R+1); \
}

//
// END SKEIN256
//

//
// BEGIN BMW
//


#define shl(x, n)            ((x) << (n))
#define shr(x, n)            ((x) >> (n))
//#define SHR(x, n) SHR2(x, n) 
//#define SHL(x, n) SHL2(x, n) 


#define ss0(x)  (shr((x), 1) ^ shl((x), 3) ^ SPH_ROTL32((x),  4) ^ SPH_ROTL32((x), 19))
#define ss1(x)  (shr((x), 1) ^ shl((x), 2) ^ SPH_ROTL32((x),  8) ^ SPH_ROTL32((x), 23))
#define ss2(x)  (shr((x), 2) ^ shl((x), 1) ^ SPH_ROTL32((x), 12) ^ SPH_ROTL32((x), 25))
#define ss3(x)  (shr((x), 2) ^ shl((x), 2) ^ SPH_ROTL32((x), 15) ^ SPH_ROTL32((x), 29))
#define ss4(x)  (shr((x), 1) ^ (x))
#define ss5(x)  (shr((x), 2) ^ (x))
#define rs1(x) SPH_ROTL32((x),  3)
#define rs2(x) SPH_ROTL32((x),  7)
#define rs3(x) SPH_ROTL32((x), 13)
#define rs4(x) SPH_ROTL32((x), 16)
#define rs5(x) SPH_ROTL32((x), 19)
#define rs6(x) SPH_ROTL32((x), 23)
#define rs7(x) SPH_ROTL32((x), 27)

/* Message expansion function 1 */
uint expand32_1(int i, const uint *M32, const uint *H, const uint *Q)
{

	return (ss1(Q[i - 16]) + ss2(Q[i - 15]) + ss3(Q[i - 14]) + ss0(Q[i - 13])
		+ ss1(Q[i - 12]) + ss2(Q[i - 11]) + ss3(Q[i - 10]) + ss0(Q[i - 9])
		+ ss1(Q[i - 8]) + ss2(Q[i - 7]) + ss3(Q[i - 6]) + ss0(Q[i - 5])
		+ ss1(Q[i - 4]) + ss2(Q[i - 3]) + ss3(Q[i - 2]) + ss0(Q[i - 1])
		+ ((i*(0x05555555ul) + SPH_ROTL32(M32[(i - 16) % 16], ((i - 16) % 16) + 1) + SPH_ROTL32(M32[(i - 13) % 16], ((i - 13) % 16) + 1) - SPH_ROTL32(M32[(i - 6) % 16], ((i - 6) % 16) + 1)) ^ H[(i - 16 + 7) % 16]));

}

/* Message expansion function 2 */
uint expand32_2(int i, const uint *M32, const uint *H, const uint *Q)
{

	return (Q[i - 16] + rs1(Q[i - 15]) + Q[i - 14] + rs2(Q[i - 13])
		+ Q[i - 12] + rs3(Q[i - 11]) + Q[i - 10] + rs4(Q[i - 9])
		+ Q[i - 8] + rs5(Q[i - 7]) + Q[i - 6] + rs6(Q[i - 5])
		+ Q[i - 4] + rs7(Q[i - 3]) + ss4(Q[i - 2]) + ss5(Q[i - 1])
		+ ((i*(0x05555555ul) + SPH_ROTL32(M32[(i - 16) % 16], ((i - 16) % 16) + 1) + SPH_ROTL32(M32[(i - 13) % 16], ((i - 13) % 16) + 1) - SPH_ROTL32(M32[(i - 6) % 16], ((i - 6) % 16) + 1)) ^ H[(i - 16 + 7) % 16]));

}

void Compression256(const uint *M32, uint *H)
{
	int i;
	uint XL32, XH32, Q[32];


	Q[0] = (M32[5] ^ H[5]) - (M32[7] ^ H[7]) + (M32[10] ^ H[10]) + (M32[13] ^ H[13]) + (M32[14] ^ H[14]);
	Q[1] = (M32[6] ^ H[6]) - (M32[8] ^ H[8]) + (M32[11] ^ H[11]) + (M32[14] ^ H[14]) - (M32[15] ^ H[15]);
	Q[2] = (M32[0] ^ H[0]) + (M32[7] ^ H[7]) + (M32[9] ^ H[9]) - (M32[12] ^ H[12]) + (M32[15] ^ H[15]);
	Q[3] = (M32[0] ^ H[0]) - (M32[1] ^ H[1]) + (M32[8] ^ H[8]) - (M32[10] ^ H[10]) + (M32[13] ^ H[13]);
	Q[4] = (M32[1] ^ H[1]) + (M32[2] ^ H[2]) + (M32[9] ^ H[9]) - (M32[11] ^ H[11]) - (M32[14] ^ H[14]);
	Q[5] = (M32[3] ^ H[3]) - (M32[2] ^ H[2]) + (M32[10] ^ H[10]) - (M32[12] ^ H[12]) + (M32[15] ^ H[15]);
	Q[6] = (M32[4] ^ H[4]) - (M32[0] ^ H[0]) - (M32[3] ^ H[3]) - (M32[11] ^ H[11]) + (M32[13] ^ H[13]);
	Q[7] = (M32[1] ^ H[1]) - (M32[4] ^ H[4]) - (M32[5] ^ H[5]) - (M32[12] ^ H[12]) - (M32[14] ^ H[14]);
	Q[8] = (M32[2] ^ H[2]) - (M32[5] ^ H[5]) - (M32[6] ^ H[6]) + (M32[13] ^ H[13]) - (M32[15] ^ H[15]);
	Q[9] = (M32[0] ^ H[0]) - (M32[3] ^ H[3]) + (M32[6] ^ H[6]) - (M32[7] ^ H[7]) + (M32[14] ^ H[14]);
	Q[10] = (M32[8] ^ H[8]) - (M32[1] ^ H[1]) - (M32[4] ^ H[4]) - (M32[7] ^ H[7]) + (M32[15] ^ H[15]);
	Q[11] = (M32[8] ^ H[8]) - (M32[0] ^ H[0]) - (M32[2] ^ H[2]) - (M32[5] ^ H[5]) + (M32[9] ^ H[9]);
	Q[12] = (M32[1] ^ H[1]) + (M32[3] ^ H[3]) - (M32[6] ^ H[6]) - (M32[9] ^ H[9]) + (M32[10] ^ H[10]);
	Q[13] = (M32[2] ^ H[2]) + (M32[4] ^ H[4]) + (M32[7] ^ H[7]) + (M32[10] ^ H[10]) + (M32[11] ^ H[11]);
	Q[14] = (M32[3] ^ H[3]) - (M32[5] ^ H[5]) + (M32[8] ^ H[8]) - (M32[11] ^ H[11]) - (M32[12] ^ H[12]);
	Q[15] = (M32[12] ^ H[12]) - (M32[4] ^ H[4]) - (M32[6] ^ H[6]) - (M32[9] ^ H[9]) + (M32[13] ^ H[13]);

	/*  Diffuse the differences in every word in a bijective manner with ssi, and then add the values of the previous double pipe.*/
	Q[0] = ss0(Q[0]) + H[1];
	Q[1] = ss1(Q[1]) + H[2];
	Q[2] = ss2(Q[2]) + H[3];
	Q[3] = ss3(Q[3]) + H[4];
	Q[4] = ss4(Q[4]) + H[5];
	Q[5] = ss0(Q[5]) + H[6];
	Q[6] = ss1(Q[6]) + H[7];
	Q[7] = ss2(Q[7]) + H[8];
	Q[8] = ss3(Q[8]) + H[9];
	Q[9] = ss4(Q[9]) + H[10];
	Q[10] = ss0(Q[10]) + H[11];
	Q[11] = ss1(Q[11]) + H[12];
	Q[12] = ss2(Q[12]) + H[13];
	Q[13] = ss3(Q[13]) + H[14];
	Q[14] = ss4(Q[14]) + H[15];
	Q[15] = ss0(Q[15]) + H[0];

	/* This is the Message expansion or f_1 in the documentation.       */
	/* It has 16 rounds.                                                */
	/* Blue Midnight Wish has two tunable security parameters.          */
	/* The parameters are named EXPAND_1_ROUNDS and EXPAND_2_ROUNDS.    */
	/* The following relation for these parameters should is satisfied: */
	/* EXPAND_1_ROUNDS + EXPAND_2_ROUNDS = 16                           */
#pragma unroll
	for (i = 0; i<2; i++)
		Q[i + 16] = expand32_1(i + 16, M32, H, Q);

#pragma unroll
	for (i = 2; i<16; i++)
		Q[i + 16] = expand32_2(i + 16, M32, H, Q);

	/* Blue Midnight Wish has two temporary cummulative variables that accumulate via XORing */
	/* 16 new variables that are prooduced in the Message Expansion part.                    */
	XL32 = Q[16] ^ Q[17] ^ Q[18] ^ Q[19] ^ Q[20] ^ Q[21] ^ Q[22] ^ Q[23];
	XH32 = XL32^Q[24] ^ Q[25] ^ Q[26] ^ Q[27] ^ Q[28] ^ Q[29] ^ Q[30] ^ Q[31];


	/*  This part is the function f_2 - in the documentation            */

	/*  Compute the double chaining pipe for the next message block.    */
	H[0] = (shl(XH32, 5) ^ shr(Q[16], 5) ^ M32[0]) + (XL32    ^ Q[24] ^ Q[0]);
	H[1] = (shr(XH32, 7) ^ shl(Q[17], 8) ^ M32[1]) + (XL32    ^ Q[25] ^ Q[1]);
	H[2] = (shr(XH32, 5) ^ shl(Q[18], 5) ^ M32[2]) + (XL32    ^ Q[26] ^ Q[2]);
	H[3] = (shr(XH32, 1) ^ shl(Q[19], 5) ^ M32[3]) + (XL32    ^ Q[27] ^ Q[3]);
	H[4] = (shr(XH32, 3) ^ Q[20] ^ M32[4]) + (XL32    ^ Q[28] ^ Q[4]);
	H[5] = (shl(XH32, 6) ^ shr(Q[21], 6) ^ M32[5]) + (XL32    ^ Q[29] ^ Q[5]);
	H[6] = (shr(XH32, 4) ^ shl(Q[22], 6) ^ M32[6]) + (XL32    ^ Q[30] ^ Q[6]);
	H[7] = (shr(XH32, 11) ^ shl(Q[23], 2) ^ M32[7]) + (XL32    ^ Q[31] ^ Q[7]);

	H[8] = SPH_ROTL32(H[4], 9) + (XH32     ^     Q[24] ^ M32[8]) + (shl(XL32, 8) ^ Q[23] ^ Q[8]);
	H[9] = SPH_ROTL32(H[5], 10) + (XH32     ^     Q[25] ^ M32[9]) + (shr(XL32, 6) ^ Q[16] ^ Q[9]);
	H[10] = SPH_ROTL32(H[6], 11) + (XH32     ^     Q[26] ^ M32[10]) + (shl(XL32, 6) ^ Q[17] ^ Q[10]);
	H[11] = SPH_ROTL32(H[7], 12) + (XH32     ^     Q[27] ^ M32[11]) + (shl(XL32, 4) ^ Q[18] ^ Q[11]);
	H[12] = SPH_ROTL32(H[0], 13) + (XH32     ^     Q[28] ^ M32[12]) + (shr(XL32, 3) ^ Q[19] ^ Q[12]);
	H[13] = SPH_ROTL32(H[1], 14) + (XH32     ^     Q[29] ^ M32[13]) + (shr(XL32, 4) ^ Q[20] ^ Q[13]);
	H[14] = SPH_ROTL32(H[2], 15) + (XH32     ^     Q[30] ^ M32[14]) + (shr(XL32, 7) ^ Q[21] ^ Q[14]);
	H[15] = SPH_ROTL32(H[3], 16) + (XH32     ^     Q[31] ^ M32[15]) + (shr(XL32, 2) ^ Q[22] ^ Q[15]);

}

//
// END BMW
//


//
// BEGIN CUBEHASH
//

#if !defined SPH_CUBEHASH_UNROLL
#define SPH_CUBEHASH_UNROLL   0
#endif

__constant static const uint CUBEHASH_IV512[] = {
  SPH_C32(0x2AEA2A61), SPH_C32(0x50F494D4), SPH_C32(0x2D538B8B),
  SPH_C32(0x4167D83E), SPH_C32(0x3FEE2313), SPH_C32(0xC701CF8C),
  SPH_C32(0xCC39968E), SPH_C32(0x50AC5695), SPH_C32(0x4D42C787),
  SPH_C32(0xA647A8B3), SPH_C32(0x97CF0BEF), SPH_C32(0x825B4537),
  SPH_C32(0xEEF864D2), SPH_C32(0xF22090C4), SPH_C32(0xD0E5CD33),
  SPH_C32(0xA23911AE), SPH_C32(0xFCD398D9), SPH_C32(0x148FE485),
  SPH_C32(0x1B017BEF), SPH_C32(0xB6444532), SPH_C32(0x6A536159),
  SPH_C32(0x2FF5781C), SPH_C32(0x91FA7934), SPH_C32(0x0DBADEA9),
  SPH_C32(0xD65C8A2B), SPH_C32(0xA5A70E75), SPH_C32(0xB1C62456),
  SPH_C32(0xBC796576), SPH_C32(0x1921C8F7), SPH_C32(0xE7989AF1),
  SPH_C32(0x7795D246), SPH_C32(0xD43E3B44)
};

#define T32      SPH_T32
#define ROTL32   SPH_ROTL32

#define ROUND_EVEN   do { \
    xg = T32(x0 + xg); \
    x0 = ROTL32(x0, 7); \
    xh = T32(x1 + xh); \
    x1 = ROTL32(x1, 7); \
    xi = T32(x2 + xi); \
    x2 = ROTL32(x2, 7); \
    xj = T32(x3 + xj); \
    x3 = ROTL32(x3, 7); \
    xk = T32(x4 + xk); \
    x4 = ROTL32(x4, 7); \
    xl = T32(x5 + xl); \
    x5 = ROTL32(x5, 7); \
    xm = T32(x6 + xm); \
    x6 = ROTL32(x6, 7); \
    xn = T32(x7 + xn); \
    x7 = ROTL32(x7, 7); \
    xo = T32(x8 + xo); \
    x8 = ROTL32(x8, 7); \
    xp = T32(x9 + xp); \
    x9 = ROTL32(x9, 7); \
    xq = T32(xa + xq); \
    xa = ROTL32(xa, 7); \
    xr = T32(xb + xr); \
    xb = ROTL32(xb, 7); \
    xs = T32(xc + xs); \
    xc = ROTL32(xc, 7); \
    xt = T32(xd + xt); \
    xd = ROTL32(xd, 7); \
    xu = T32(xe + xu); \
    xe = ROTL32(xe, 7); \
    xv = T32(xf + xv); \
    xf = ROTL32(xf, 7); \
    x8 ^= xg; \
    x9 ^= xh; \
    xa ^= xi; \
    xb ^= xj; \
    xc ^= xk; \
    xd ^= xl; \
    xe ^= xm; \
    xf ^= xn; \
    x0 ^= xo; \
    x1 ^= xp; \
    x2 ^= xq; \
    x3 ^= xr; \
    x4 ^= xs; \
    x5 ^= xt; \
    x6 ^= xu; \
    x7 ^= xv; \
    xi = T32(x8 + xi); \
    x8 = ROTL32(x8, 11); \
    xj = T32(x9 + xj); \
    x9 = ROTL32(x9, 11); \
    xg = T32(xa + xg); \
    xa = ROTL32(xa, 11); \
    xh = T32(xb + xh); \
    xb = ROTL32(xb, 11); \
    xm = T32(xc + xm); \
    xc = ROTL32(xc, 11); \
    xn = T32(xd + xn); \
    xd = ROTL32(xd, 11); \
    xk = T32(xe + xk); \
    xe = ROTL32(xe, 11); \
    xl = T32(xf + xl); \
    xf = ROTL32(xf, 11); \
    xq = T32(x0 + xq); \
    x0 = ROTL32(x0, 11); \
    xr = T32(x1 + xr); \
    x1 = ROTL32(x1, 11); \
    xo = T32(x2 + xo); \
    x2 = ROTL32(x2, 11); \
    xp = T32(x3 + xp); \
    x3 = ROTL32(x3, 11); \
    xu = T32(x4 + xu); \
    x4 = ROTL32(x4, 11); \
    xv = T32(x5 + xv); \
    x5 = ROTL32(x5, 11); \
    xs = T32(x6 + xs); \
    x6 = ROTL32(x6, 11); \
    xt = T32(x7 + xt); \
    x7 = ROTL32(x7, 11); \
    xc ^= xi; \
    xd ^= xj; \
    xe ^= xg; \
    xf ^= xh; \
    x8 ^= xm; \
    x9 ^= xn; \
    xa ^= xk; \
    xb ^= xl; \
    x4 ^= xq; \
    x5 ^= xr; \
    x6 ^= xo; \
    x7 ^= xp; \
    x0 ^= xu; \
    x1 ^= xv; \
    x2 ^= xs; \
    x3 ^= xt; \
  } while (0)

#define ROUND_ODD   do { \
    xj = T32(xc + xj); \
    xc = ROTL32(xc, 7); \
    xi = T32(xd + xi); \
    xd = ROTL32(xd, 7); \
    xh = T32(xe + xh); \
    xe = ROTL32(xe, 7); \
    xg = T32(xf + xg); \
    xf = ROTL32(xf, 7); \
    xn = T32(x8 + xn); \
    x8 = ROTL32(x8, 7); \
    xm = T32(x9 + xm); \
    x9 = ROTL32(x9, 7); \
    xl = T32(xa + xl); \
    xa = ROTL32(xa, 7); \
    xk = T32(xb + xk); \
    xb = ROTL32(xb, 7); \
    xr = T32(x4 + xr); \
    x4 = ROTL32(x4, 7); \
    xq = T32(x5 + xq); \
    x5 = ROTL32(x5, 7); \
    xp = T32(x6 + xp); \
    x6 = ROTL32(x6, 7); \
    xo = T32(x7 + xo); \
    x7 = ROTL32(x7, 7); \
    xv = T32(x0 + xv); \
    x0 = ROTL32(x0, 7); \
    xu = T32(x1 + xu); \
    x1 = ROTL32(x1, 7); \
    xt = T32(x2 + xt); \
    x2 = ROTL32(x2, 7); \
    xs = T32(x3 + xs); \
    x3 = ROTL32(x3, 7); \
    x4 ^= xj; \
    x5 ^= xi; \
    x6 ^= xh; \
    x7 ^= xg; \
    x0 ^= xn; \
    x1 ^= xm; \
    x2 ^= xl; \
    x3 ^= xk; \
    xc ^= xr; \
    xd ^= xq; \
    xe ^= xp; \
    xf ^= xo; \
    x8 ^= xv; \
    x9 ^= xu; \
    xa ^= xt; \
    xb ^= xs; \
    xh = T32(x4 + xh); \
    x4 = ROTL32(x4, 11); \
    xg = T32(x5 + xg); \
    x5 = ROTL32(x5, 11); \
    xj = T32(x6 + xj); \
    x6 = ROTL32(x6, 11); \
    xi = T32(x7 + xi); \
    x7 = ROTL32(x7, 11); \
    xl = T32(x0 + xl); \
    x0 = ROTL32(x0, 11); \
    xk = T32(x1 + xk); \
    x1 = ROTL32(x1, 11); \
    xn = T32(x2 + xn); \
    x2 = ROTL32(x2, 11); \
    xm = T32(x3 + xm); \
    x3 = ROTL32(x3, 11); \
    xp = T32(xc + xp); \
    xc = ROTL32(xc, 11); \
    xo = T32(xd + xo); \
    xd = ROTL32(xd, 11); \
    xr = T32(xe + xr); \
    xe = ROTL32(xe, 11); \
    xq = T32(xf + xq); \
    xf = ROTL32(xf, 11); \
    xt = T32(x8 + xt); \
    x8 = ROTL32(x8, 11); \
    xs = T32(x9 + xs); \
    x9 = ROTL32(x9, 11); \
    xv = T32(xa + xv); \
    xa = ROTL32(xa, 11); \
    xu = T32(xb + xu); \
    xb = ROTL32(xb, 11); \
    x0 ^= xh; \
    x1 ^= xg; \
    x2 ^= xj; \
    x3 ^= xi; \
    x4 ^= xl; \
    x5 ^= xk; \
    x6 ^= xn; \
    x7 ^= xm; \
    x8 ^= xp; \
    x9 ^= xo; \
    xa ^= xr; \
    xb ^= xq; \
    xc ^= xt; \
    xd ^= xs; \
    xe ^= xv; \
    xf ^= xu; \
  } while (0)

/*
 * There is no need to unroll all 16 rounds. The word-swapping permutation
 * is an involution, so we need to unroll an even number of rounds. On
 * "big" systems, unrolling 4 rounds yields about 97% of the speed
 * achieved with full unrolling; and it keeps the code more compact
 * for small architectures.
 */

#if SPH_CUBEHASH_UNROLL == 2

#define SIXTEEN_ROUNDS   do { \
    int j; \
    for (j = 0; j < 8; j ++) { \
      ROUND_EVEN; \
      ROUND_ODD; \
    } \
  } while (0)

#elif SPH_CUBEHASH_UNROLL == 4

#define SIXTEEN_ROUNDS   do { \
    int j; \
    for (j = 0; j < 4; j ++) { \
      ROUND_EVEN; \
      ROUND_ODD; \
      ROUND_EVEN; \
      ROUND_ODD; \
    } \
  } while (0)

#elif SPH_CUBEHASH_UNROLL == 8

#define SIXTEEN_ROUNDS   do { \
    int j; \
    for (j = 0; j < 2; j ++) { \
      ROUND_EVEN; \
      ROUND_ODD; \
      ROUND_EVEN; \
      ROUND_ODD; \
      ROUND_EVEN; \
      ROUND_ODD; \
      ROUND_EVEN; \
      ROUND_ODD; \
    } \
  } while (0)

#else

#define SIXTEEN_ROUNDS   do { \
    ROUND_EVEN; \
    ROUND_ODD; \
    ROUND_EVEN; \
    ROUND_ODD; \
    ROUND_EVEN; \
    ROUND_ODD; \
    ROUND_EVEN; \
    ROUND_ODD; \
    ROUND_EVEN; \
    ROUND_ODD; \
    ROUND_EVEN; \
    ROUND_ODD; \
    ROUND_EVEN; \
    ROUND_ODD; \
    ROUND_EVEN; \
    ROUND_ODD; \
  } while (0)

#endif

//
// END CUBEHASH
//

//
// BEGIN LYRA
//


// LYRA2 PREPROCESSOR MACROS


// replicate build env
#define LYRA_SCRATCHBUF_SIZE 1536
#define memshift 3


// opencl versions
#define ROTL64(x,n) rotate(x,(ulong)n)
#define ROTR64(x,n) rotate(x,(ulong)(64-n))
#define SWAP32(x) as_ulong(as_uint2(x).s10)

//#define ROTL64(x,n) SPH_ROTL64(x, n)
//#define ROTR64(x,n) SPH_ROTR64(x, n)
//#define SWAP32(x) sph_bswap32(x)


/*One Round of the Blake2b's compression function*/

#define G_old(a,b,c,d) \
  do { \
	a += b; d ^= a; d = ROTR64(d, 32); \
	c += d; b ^= c; b = ROTR64(b, 24); \
	a += b; d ^= a; d = ROTR64(d, 16); \
	c += d; b ^= c; b = ROTR64(b, 63); \
\
  } while (0)

#define round_lyra(s)  \
 do { \
	 G_old(s[0].x, s[1].x, s[2].x, s[3].x); \
     G_old(s[0].y, s[1].y, s[2].y, s[3].y); \
     G_old(s[0].z, s[1].z, s[2].z, s[3].z); \
     G_old(s[0].w, s[1].w, s[2].w, s[3].w); \
     G_old(s[0].x, s[1].y, s[2].z, s[3].w); \
     G_old(s[0].y, s[1].z, s[2].w, s[3].x); \
     G_old(s[0].z, s[1].w, s[2].x, s[3].y); \
     G_old(s[0].w, s[1].x, s[2].y, s[3].z); \
 } while(0)

#define G(a,b,c,d) \
  do { \
	a += b; d ^= a; d = SWAP32(d); \
	c += d; b ^= c; b = round_lyra(b,24); \
	a += b; d ^= a; d = ROTR64(d,16); \
	c += d; b ^= c; b = ROTR64(b, 63); \
\
  } while (0)

#define SPH_ULONG4(a, b, c, d) (ulong4)(a, b, c, d)


void reduceDuplexf(ulong4* state , __global ulong4* DMatrix)
{

	 ulong4 state1[3];
	 uint ps1 = 0;
	 uint ps2 = (memshift * 3 + memshift * 4);
//#pragma unroll 4
	 for (int i = 0; i < 4; i++)
	 {
		 uint s1 = ps1 + i*memshift;
		 uint s2 = ps2 - i*memshift;

		 for (int j = 0; j < 3; j++)  state1[j] = (DMatrix)[j + s1];

		 for (int j = 0; j < 3; j++)  state[j] ^= state1[j];
		 round_lyra(state);
		 for (int j = 0; j < 3; j++)  state1[j] ^= state[j];

		 for (int j = 0; j < 3; j++)  (DMatrix)[j + s2] = state1[j];
	 }

}

void reduceDuplexRowf(uint rowIn,uint rowInOut,uint rowOut,ulong4 * state, __global ulong4 * DMatrix)
{

ulong4 state1[3], state2[3];
uint ps1 = (memshift * 4 * rowIn);
uint ps2 = (memshift * 4 * rowInOut);
uint ps3 = (memshift * 4 * rowOut);


  for (int i = 0; i < 4; i++)
 {
  uint s1 = ps1 + i*memshift;
  uint s2 = ps2 + i*memshift;
  uint s3 = ps3 + i*memshift;


		 for (int j = 0; j < 3; j++)   state1[j] = (DMatrix)[j + s1];

         for (int j = 0; j < 3; j++)   state2[j] = (DMatrix)[j + s2];

         for (int j = 0; j < 3; j++)   state1[j] += state2[j];

         for (int j = 0; j < 3; j++)   state[j] ^= state1[j];


         round_lyra(state);

         ((ulong*)state2)[0] ^= ((ulong*)state)[11];
  for (int j = 0; j < 11; j++)
	  ((ulong*)state2)[j + 1] ^= ((ulong*)state)[j];

         if (rowInOut != rowOut) {
			 for (int j = 0; j < 3; j++)
				 (DMatrix)[j + s2] = state2[j];
			 for (int j = 0; j < 3; j++)
				 (DMatrix)[j + s3] ^= state[j];
  		 }
		 else {
			 for (int j = 0; j < 3; j++)
				 state2[j] ^= state[j];
			 for (int j = 0; j < 3; j++)
				 (DMatrix)[j + s2] = state2[j];
		 }

 }
  }




void reduceDuplexRowSetupf(uint rowIn, uint rowInOut, uint rowOut, ulong4 *state,  __global ulong4* DMatrix) {

	 ulong4 state2[3], state1[3];
	 uint ps1 = (memshift * 4 * rowIn);
	 uint ps2 = (memshift * 4 * rowInOut);
	 uint ps3 = (memshift * 3 + memshift * 4 * rowOut);

	 for (int i = 0; i < 4; i++)
	 {
		 uint s1 = ps1 + i*memshift;
		 uint s2 = ps2 + i*memshift;
		 uint s3 = ps3 - i*memshift;

		 for (int j = 0; j < 3; j++)  state1[j] = (DMatrix)[j + s1];

		 for (int j = 0; j < 3; j++)  state2[j] = (DMatrix)[j + s2];
		 for (int j = 0; j < 3; j++) {
			 ulong4 tmp = state1[j] + state2[j];
			 state[j] ^= tmp;
		 		 }
		 round_lyra(state);

		 for (int j = 0; j < 3; j++) {
			 state1[j] ^= state[j];
			 (DMatrix)[j + s3] = state1[j];
		 		 }

		 ((ulong*)state2)[0] ^= ((ulong*)state)[11];
		 for (int j = 0; j < 11; j++)
			 ((ulong*)state2)[j + 1] ^= ((ulong*)state)[j];
		 for (int j = 0; j < 3; j++)
			 (DMatrix)[j + s2] = state2[j];
	 }
}

// END LYRA2 PREPROCESSOR MACROS

//
// END LYRA
//


//#define DEBUG


// Hash helper functions

// blake80, in(80 bytes), out(32 bytes)
void blake80_noswap(const uint* input_words, uint* out_words)
{;
	//printf("INPUT WORDS[1]: %s\n", debug_print_hash((uint*)(input_words)));
	//printf("INPUT WORDS[2]: %s\n", debug_print_hash((uint*)((uint8_t*)(input_words) + (52 - 32))));
	
	// Blake256 vars
	BLAKE256_STATE;
	BLAKE256_COMPRESS32_STATE;

	// Blake256 start hash
	INIT_BLAKE256_STATE;
	// Blake hash full input
	T0 = SPH_T32(T0 + 512);

	//printf("blake32 full step T0=0x%x T1=0x%x H=[%x,%x,%x,%x,%x,%x,%x,%x] S=[%x,%x,%x,%x]\n", T0, T1, H0, H1, H2, H3, H4, H5, H6, H7, S0, S1, S2, S3);

	BLAKE256_COMPRESS32((input_words[0]),(input_words[1]),(input_words[2]),(input_words[3]),(input_words[4]),(input_words[5]),(input_words[6]),(input_words[7]),(input_words[8]),(input_words[9]),(input_words[10]),(input_words[11]),(input_words[12]),(input_words[13]),(input_words[14]),(input_words[15]));
	
	//printf("blake32 after step T0=0x%x T1=0x%x H=[%x,%x,%x,%x,%x,%x,%x,%x] S=[%x,%x,%x,%x]\n", T0, T1, H0, H1, H2, H3, H4, H5, H6, H7, S0, S1, S2, S3);

	// blake close - filled case
	T0 -= 512 - 128; // i.e. 128
	T0 = SPH_T32(T0 + 512);

	//printf("blake32 full step T0=0x%x T1=0x%x H=[%x,%x,%x,%x,%x,%x,%x,%x] S=[%x,%x,%x,%x]\n", T0, T1, H0, H1, H2, H3, H4, H5, H6, H7, S0, S1, S2, S3);

	BLAKE256_COMPRESS32((input_words[16]),(input_words[17]),(input_words[18]),(input_words[19]),2147483648,0,0,0,0,0,0,0,0,1,0,640);
	// output to BLAKE_OUT_HASH
	out_words[0] = sph_bswap32(H0);
	out_words[1] = sph_bswap32(H1);
	out_words[2] = sph_bswap32(H2);
	out_words[3] = sph_bswap32(H3);
	out_words[4] = sph_bswap32(H4);
	out_words[5] = sph_bswap32(H5);
	out_words[6] = sph_bswap32(H6);
	out_words[7] = sph_bswap32(H7);
}

// blake80, in(52 bytes), out(32 bytes)
void blake52(const uint* input_words, uint* out_words)
{
	// Blake256 vars
	BLAKE256_STATE;
	BLAKE256_COMPRESS32_STATE;

	// Blake256 start hash
	INIT_BLAKE256_STATE;
	// Blake hash full input
	// blake close - t0==0 case
	T0 = SPH_C32(0xFFFFFE00) + 416;
	T1 = SPH_C32(0xFFFFFFFF);
	T0 = SPH_T32(T0 + 512);
	T1 = SPH_T32(T1 + 1);

	//printf("blake32 full step T0=0x%x T1=0x%x H=[%x,%x,%x,%x,%x,%x,%x,%x] S=[%x,%x,%x,%x]\n", T0, T1, H0, H1, H2, H3, H4, H5, H6, H7, S0, S1, S2, S3);

	BLAKE256_COMPRESS32(sph_bswap32(input_words[0]),sph_bswap32(input_words[1]),sph_bswap32(input_words[2]),sph_bswap32(input_words[3]),sph_bswap32(input_words[4]),sph_bswap32(input_words[5]),sph_bswap32(input_words[6]),sph_bswap32(input_words[7]),sph_bswap32(input_words[8]),sph_bswap32(input_words[9]),sph_bswap32(input_words[10]),sph_bswap32(input_words[11]),sph_bswap32(input_words[12]),2147483649,0,416);

	//printf("blake32 final step T0=0x%x T1=0x%x H=[%x,%x,%x,%x,%x,%x,%x,%x] S=[%x,%x,%x,%x]\n", T0, T1, H0, H1, H2, H3, H4, H5, H6, H7, S0, S1, S2, S3);

	out_words[0] = sph_bswap32(H0);
	out_words[1] = sph_bswap32(H1);
	out_words[2] = sph_bswap32(H2);
	out_words[3] = sph_bswap32(H3);
	out_words[4] = sph_bswap32(H4);
	out_words[5] = sph_bswap32(H5);
	out_words[6] = sph_bswap32(H6);
	out_words[7] = sph_bswap32(H7);
}


// skein32, in(32 bytes), out(32 bytes)
void skein32(const ulong* in_dwords, ulong* out_dwords)
{
	//printf("skein32 in=%s\n", debug_print_hash(in_dwords));

	ulong h[9];
	ulong t[3];
	ulong dt0,dt1,dt2,dt3;
	ulong p0, p1, p2, p3, p4, p5, p6, p7;
	h[8] = skein_ks_parity;

	for (int i = 0; i<8; i++) {
		h[i] = SKEIN_IV512_256[i];
		h[8] ^= h[i];
	}

	t[0]=t12[0];
	t[1]=t12[1];
	t[2]=t12[2];

	dt0= (in_dwords[0]);
	dt1= (in_dwords[1]);
	dt2= (in_dwords[2]);
	dt3= (in_dwords[3]);

	//printf("Skein in hash=%lu,%lu,%lu,%lu\n",
	//	dt0, dt1, dt2, dt3);

	p0 = h[0] + dt0;
	p1 = h[1] + dt1;
	p2 = h[2] + dt2;
	p3 = h[3] + dt3;
	p4 = h[4];
	p5 = h[5] + t[0];
	p6 = h[6] + t[1];
	p7 = h[7];

	#pragma unroll 
	for (int i = 1; i<19; i+=2) {Round_8_512(p0,p1,p2,p3,p4,p5,p6,p7,i);}
	
	p0 ^= dt0;
	p1 ^= dt1;
	p2 ^= dt2;
	p3 ^= dt3;

	h[0] = p0;
	h[1] = p1;
	h[2] = p2;
	h[3] = p3;
	h[4] = p4;
	h[5] = p5;
	h[6] = p6;
	h[7] = p7;
	h[8] = skein_ks_parity;

	for (int i = 0; i<8; i++) { h[8] ^= h[i]; }
		
	t[0] = t12[3];
	t[1] = t12[4];
	t[2] = t12[5];
	p5 += t[0];  //p5 already equal h[5] 
	p6 += t[1];

        #pragma unroll
	for (int i = 1; i<19; i+=2) { Round_8_512(p0, p1, p2, p3, p4, p5, p6, p7, i); }


	//printf("skein out regs =%lu,%lu,%lu,%lu,%lu,%lu,%lu,%lu\n", p0, p1, p2, p3, p4, p5, p6, p7);

	out_dwords[0]      = (p0);
	out_dwords[1]      = (p1);
	out_dwords[2]      = (p2);
	out_dwords[3]      = (p3);
}


// keccak32, in(32 bytes), out(32 bytes)
void keccak32(const ulong* input_dwords, ulong* output_dwords)
{
	ulong keccak_gpu_state[25];

	for (int i = 0; i<25; i++) {
		if (i<4) { keccak_gpu_state[i] = (input_dwords[i]); }
		else    { keccak_gpu_state[i] = 0; }
	}
	keccak_gpu_state[4] = 0x0000000000000001;
	keccak_gpu_state[16] = 0x8000000000000000;

	keccak_block((ulong*)&keccak_gpu_state[0]);
	for (int i = 0; i<4; i++) { output_dwords[i] = keccak_gpu_state[i]; }
}

// bmw32, in(32 bytes), out(32 bytes)
void bmw32(const uint* in_words, uint* out_words)
{
	//printf("bmw32 in bytes=%s\n", debug_print_hash(in_hash));

	uint dh[16] = {
		0x40414243, 0x44454647,
		0x48494A4B, 0x4C4D4E4F,
		0x50515253, 0x54555657,
		0x58595A5B, 0x5C5D5E5F,
		0x60616263, 0x64656667,
		0x68696A6B, 0x6C6D6E6F,
		0x70717273, 0x74757677,
		0x78797A7B, 0x7C7D7E7F
	};
	uint final_s[16] = {
		0xaaaaaaa0, 0xaaaaaaa1, 0xaaaaaaa2,
		0xaaaaaaa3, 0xaaaaaaa4, 0xaaaaaaa5,
		0xaaaaaaa6, 0xaaaaaaa7, 0xaaaaaaa8,
		0xaaaaaaa9, 0xaaaaaaaa, 0xaaaaaaab,
		0xaaaaaaac, 0xaaaaaaad, 0xaaaaaaae,
		0xaaaaaaaf
	};

	uint message[16];
	for (int i = 0; i<8; i++) message[i] = (in_words[i]);
	for (int i = 9; i<14; i++) message[i] = 0;
	message[8]= 0x80;
	message[14]=0x100;
	message[15]=0;

	Compression256(message, dh);
	Compression256(dh, final_s);

	#pragma unroll
	for (int i=8; i<16; i++) {
		out_words[i-8] = (final_s[i]);
	}
}

// cubehash32, in(32 bytes), out(32 bytes)
void cubehash32(const uint* in_words, uint* out_words)
{
	uint x0 = 0xEA2BD4B4; uint x1 = 0xCCD6F29F; uint x2 = 0x63117E71;
	uint x3 = 0x35481EAE; uint x4 = 0x22512D5B; uint x5 = 0xE5D94E63;
	uint x6 = 0x7E624131; uint x7 = 0xF4CC12BE; uint x8 = 0xC2D0B696;
	uint x9 = 0x42AF2070; uint xa = 0xD0720C35; uint xb = 0x3361DA8C;
	uint xc = 0x28CCECA4; uint xd = 0x8EF8AD83; uint xe = 0x4680AC00;
	uint xf = 0x40E5FBAB;

	uint xg = 0xD89041C3; uint xh = 0x6107FBD5;
	uint xi = 0x6C859D41; uint xj = 0xF0B26679; uint xk = 0x09392549;
	uint xl = 0x5FA25603; uint xm = 0x65C892FD; uint xn = 0x93CB6285;
	uint xo = 0x2AF2B5AE; uint xp = 0x9E4B4E60; uint xq = 0x774ABFDD;
	uint xr = 0x85254725; uint xs = 0x15815AEB; uint xt = 0x4AB6AAD6;
	uint xu = 0x9CDAF8AF; uint xv = 0xD6032C0A;

	x0 ^= (in_words[0]);
	x1 ^= (in_words[1]);
	x2 ^= (in_words[2]);
	x3 ^= (in_words[3]);
	x4 ^= (in_words[4]);
	x5 ^= (in_words[5]);
	x6 ^= (in_words[6]);
	x7 ^= (in_words[7]);


	SIXTEEN_ROUNDS;
	x0 ^= 0x80;
	SIXTEEN_ROUNDS;
	xv ^= 0x01;
	for (int i = 0; i < 10; ++i) SIXTEEN_ROUNDS;

	out_words[0] = x0;
	out_words[1] = x1;
	out_words[2] = x2;
	out_words[3] = x3;
	out_words[4] = x4;
	out_words[5] = x5;
	out_words[6] = x6;
	out_words[7] = x7;
}

// run-of-the-mill lyra2
void lyra2(const ulong* in_dwords, ulong* out_dwords,__global ulong4* DMatrix)
{
	ulong4 state[4];

	state[0].x = in_dwords[0]; //password
	state[0].y = in_dwords[1]; //password
	state[0].z = in_dwords[2]; //password
	state[0].w = in_dwords[3]; //password
	state[1] = state[0];
	state[2] = SPH_ULONG4(0x6a09e667f3bcc908UL, 0xbb67ae8584caa73bUL, 0x3c6ef372fe94f82bUL, 0xa54ff53a5f1d36f1UL);
	state[3] = SPH_ULONG4(0x510e527fade682d1UL, 0x9b05688c2b3e6c1fUL, 0x1f83d9abfb41bd6bUL, 0x5be0cd19137e2179UL);
	for (int i = 0; i<12; i++) { round_lyra(state); } 

	state[0] ^= SPH_ULONG4(0x20,0x20,0x20,0x01);
	state[1] ^= SPH_ULONG4(0x04,0x04,0x80,0x0100000000000000);

	for (int i = 0; i<12; i++) { round_lyra(state); } 


	uint ps1 = (memshift * 3);
	//#pragma unroll 4
	for (int i = 0; i < 4; i++)
	{
		uint s1 = ps1 - memshift * i;
		for (int j = 0; j < 3; j++)
			(DMatrix)[j+s1] = state[j];

		round_lyra(state);
	}

	reduceDuplexf(state,DMatrix);

	reduceDuplexRowSetupf(1, 0, 2,state, DMatrix);
	reduceDuplexRowSetupf(2, 1, 3, state,DMatrix);


	uint rowa;
	uint prev = 3;
	for (uint i = 0; i<4; i++) {
		rowa = state[0].x & 3;
		reduceDuplexRowf(prev, rowa, i, state, DMatrix);
		prev = i;
	}

	uint shift = (memshift * 4 * rowa);

	for (int j = 0; j < 3; j++)
		state[j] ^= (DMatrix)[j+shift];

	for (int i = 0; i < 12; i++)
		round_lyra(state);
	
	//////////////////////////////////////

	for (int i = 0; i<4; i++) {out_dwords[i] = ((ulong*)state)[i];} 
}


// lyra2re252, in(80 bytes), out(32 bytes)
void lyra2re2_hash80_noswap(const uint* blockToHash, uint* hashedHeader, __global ulong4* nodes)
{
	uint hashB[8];

    blake80_noswap(blockToHash, hashedHeader);
    keccak32((ulong*)hashedHeader, (ulong*)hashB);
    cubehash32(hashB, hashedHeader);
    lyra2((ulong*)hashedHeader, (ulong*)hashB, nodes);
    skein32((ulong*)hashB, (ulong*)hashedHeader);
    cubehash32(hashedHeader, hashB);
    bmw32(hashB, hashedHeader);
}

// lyra2re252, in(52 bytes), out(32 bytes)
void lyra2re2_hash52(const uint* blockToHash, uint* hashedHeader, __global ulong4* nodes)
{
	uint hashB[8];

    blake52(blockToHash, hashedHeader);
    keccak32((ulong*)hashedHeader, (ulong*)hashB);
    cubehash32(hashB, hashedHeader);
    lyra2((ulong*)hashedHeader, (ulong*)hashB, nodes);
    skein32((ulong*)hashB, (ulong*)hashedHeader);
    cubehash32(hashedHeader, hashB);
    bmw32(hashB, hashedHeader);
}

#define HASH_WORDS 8
#define MIX_WORDS 16

// Main hash function, in(80 bytes), out(32 bytes)
void hashimoto(uint *blockToHash, __global const uint *dag, const ulong n, const uint height, __global ulong4* nodes)
{
    const ulong mixhashes = MIX_BYTES / HASH_BYTES;
    const ulong wordhashes = MIX_BYTES / WORD_BYTES;
    uint mix[MIX_BYTES/sizeof(uint)];
    uint newdata[MIX_BYTES/sizeof(uint)];

    //uint nonce = sph_bswap32(blockToHash[19]);
	
	lyra2re2_hash80_noswap(blockToHash,blockToHash, nodes);

    //printf("nonce(%u) -> %08x%08x%08x%08x%08x%08x%08x%08x\n", nonce_debug, blockToHash[0], blockToHash[1], blockToHash[2], blockToHash[3], blockToHash[4], blockToHash[5], blockToHash[6], blockToHash[7]);

	mix[0] = blockToHash[0];
	mix[1] = blockToHash[1];
	mix[2] = blockToHash[2];
	mix[3] = blockToHash[3];
	mix[4] = blockToHash[4];
	mix[5] = blockToHash[5];
	mix[6] = blockToHash[6];
	mix[7] = blockToHash[7];
	mix[8] = blockToHash[0];
	mix[9] = blockToHash[1];
	mix[10] = blockToHash[2];
	mix[11] = blockToHash[3];
	mix[12] = blockToHash[4];
	mix[13] = blockToHash[5];
	mix[14] = blockToHash[6];
	mix[15] = blockToHash[7];

    for(int i = 0; i < ACCESSES; i++) {
        uint p = fnv(i ^ blockToHash[0], mix[i % (MIX_BYTES/sizeof(uint))]) % (n / mixhashes) * mixhashes;
        for(int j = 0; j < mixhashes; j++) {
            ulong pj = (p+j)*8;

            newdata[(j * 8)] = dag[pj];
            newdata[(j * 8)+1] = dag[pj+1];
            newdata[(j * 8)+2] = dag[pj+2];
            newdata[(j * 8)+3] = dag[pj+3];
            newdata[(j * 8)+4] = dag[pj+4];
            newdata[(j * 8)+5] = dag[pj+5];
            newdata[(j * 8)+6] = dag[pj+6];
            newdata[(j * 8)+7] = dag[pj+7];
        }
        for(int i = 0; i < MIX_BYTES/sizeof(uint); i++) {
            mix[i] = fnv(mix[i], newdata[i]);
        }
    }

    // cmix -> result.cmix. Also goes at end of header.
    blockToHash[8] = height;
    blockToHash[9] = fnv(fnv(fnv(mix[0], mix[0+1]), mix[0+2]), mix[0+3]);
	blockToHash[10] = fnv(fnv(fnv(mix[4], mix[4+1]), mix[4+2]), mix[4+3]);
	blockToHash[11] = fnv(fnv(fnv(mix[8], mix[8+1]), mix[8+2]), mix[8+3]);
	blockToHash[12] = fnv(fnv(fnv(mix[12], mix[12+1]), mix[12+2]), mix[12+3]);

    // Final hash is first hash + mix + height
    lyra2re2_hash52(blockToHash, blockToHash, nodes);
}

// Set to enable hash testing kernel variant
//#define TEST_KERNEL_HASH

#ifndef TEST_KERNEL_HASH

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search(
	__global volatile uint* restrict g_output,
	__constant uint const* g_header,
	__global uint const* g_dag,
	__global ulong4* g_lyre_nodes,
	const ulong DAG_ITEM_COUNT,
	const uint height,
	const uint target
	)
{
	const uint gid = get_global_id(0); // i.e. nonce
	const uint hash_output_idx = gid - get_global_offset(0);
	__global ulong4 *DMatrix = (__global ulong4 *)(g_lyre_nodes + (4 * memshift * 4 * 4 * 8 * (hash_output_idx % MAX_GLOBAL_THREADS)));

	uint saved_height = height;
	ulong saved_target = target;
	//printf("Search nonce %u (hash id %u) height %u target=%lx\n", gid, hash_output_idx, saved_height, saved_target);

	// set block
	uint block[20];
	block[0] = g_header[0];
	block[1] = g_header[1];
	block[2] = g_header[2];
	block[3] = g_header[3];
	block[4] = g_header[4];
	block[5] = g_header[5];
	block[6] = g_header[6];
	block[7] = g_header[7];
	block[8] = g_header[8];
	block[9] = g_header[9];
	block[10] = g_header[10];
	block[11] = g_header[11];
	block[12] = g_header[12];
	block[13] = g_header[13];
	block[14] = g_header[14];
	block[15] = g_header[15];
	block[16] = g_header[16];
	block[17] = g_header[17];
	block[18] = g_header[18];
	block[19] = gid;

	// Run hashimoto (result hash output to block)
	hashimoto(block, g_dag, DAG_ITEM_COUNT, height, DMatrix);

	//printf("NONCE[%u] TARGET %08x Hashimoto(%u, %u) -> %08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n", gid, target, DAG_ITEM_COUNT*32, height, block[0], block[1], block[2], block[3], block[4], block[5], block[6], block[7]);

	// Check target
	//ulong* out_long = &block[0];
	//out_long[3] = 0;

	// target itself should be in little-endian format, 
#ifdef NVIDIA
	if (block[7] <= target)
	{
		//printf("Nonce %u Found target, %lx <= %lx\n", gid, out_long[3], target);
		uint slot = atomic_inc(&g_output[MAX_OUTPUTS]);
		//uint2 tgt = as_uint2(target);
		//printf("candidate %u => %08x %08x < %08x\n", slot, state[0].x, state[0].y, (uint) (target>>32));
		g_output[slot & MAX_OUTPUTS] = gid;
	}
#else
	if (block[7] <= target)
	{
		//printf("Nonce %u Found target, %lx <= %lx\n", gid, out_long[3], target);
		uint slot = min(MAX_OUTPUTS-1u, convert_uint(atomic_inc(&g_output[MAX_OUTPUTS])));
		g_output[slot] = gid;
	}
#endif
}

#else

__attribute__((reqd_work_group_size(WORKSIZE, 1, 1)))
__kernel void search(
	__global volatile hash32_t* restrict g_output,
	__constant uint const* g_header,
	__global uint const* g_dag,
	__global ulong4* g_lyre_nodes,
	const ulong DAG_ITEM_COUNT,
	const uint height
	)
{
	const uint gid = get_global_id(0); // i.e. nonce
	const uint hash_output_idx = gid - get_global_offset(0);
	__global ulong4 *DMatrix = (__global ulong4 *)(g_lyre_nodes + (4 * memshift * 4 * 4 * 8 * (hash_output_idx % MAX_GLOBAL_THREADS)));

	//printf("Search nonce %u (hash id %u) DAG_ITEM_COUNT %u, height %u\n", gid, hash_output_idx, DAG_ITEM_COUNT, height);

	// set block
	uint block[20];
	block[0] = g_header[0];
	block[1] = g_header[1];
	block[2] = g_header[2];
	block[3] = g_header[3];
	block[4] = g_header[4];
	block[5] = g_header[5];
	block[6] = g_header[6];
	block[7] = g_header[7];
	block[8] = g_header[8];
	block[9] = g_header[9];
	block[10] = g_header[10];
	block[11] = g_header[11];
	block[12] = g_header[12];
	block[13] = g_header[13];
	block[14] = g_header[14];
	block[15] = g_header[15];
	block[16] = g_header[16];
	block[17] = g_header[17];
	block[18] = g_header[18];
	block[19] = gid;

	//printf("NONCE[%u] HEADER == %08x,%08x,%08x,%08x,%08x,%08x,%08x\n", gid, 
	//	block[0], block[1], block[2], block[3], block[4], block[5], block[6], block[7]);

	// Run hashimoto (result hash output to block)
	hashimoto(block, g_dag, DAG_ITEM_COUNT, height, DMatrix);

	//printf("NONCE[%u] Hashimoto(%u, %u) -> %08x,%08x,%08x,%08x,%08x,%08x,%08x,%08x\n", gid, DAG_ITEM_COUNT*32, height, block[0], block[1], block[2], block[3], block[4], block[5], block[6], block[7]);
    

	g_output[hash_output_idx].h4[0] = block[0];
	g_output[hash_output_idx].h4[1] = block[1];
	g_output[hash_output_idx].h4[2] = block[2];
	g_output[hash_output_idx].h4[3] = block[3];
	g_output[hash_output_idx].h4[4] = block[4];
	g_output[hash_output_idx].h4[5] = block[5];
	g_output[hash_output_idx].h4[6] = block[6];
	g_output[hash_output_idx].h4[7] = block[7];
}

#endif

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
