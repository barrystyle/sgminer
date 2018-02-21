#include <stdint.h>

#include "config.h"
#include "miner.h"
#include "util.h"
#include "algorithm/ethash.h"
#include "algorithm/nightcap.h"
#include "sph/sph_blake.h"
#include "sph/sph_bmw.h"
#include "sph/sph_skein.h"
#include "sph/sph_cubehash.h"
#include "sph/sph_keccak.h"
#include "algorithm/lyra2.h"


#ifdef _MSC_VER
#define restrict __restrict
#endif

// NOTE: reusing eth cache locks

extern cglock_t EthCacheLock[2];
extern uint8_t* EthCache[2];

extern pthread_mutex_t eth_nonce_lock;
extern uint32_t eth_nonce;

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
#define FNV_PRIME 0x01000193


unsigned int fnv(unsigned int v1, unsigned int v2) {
	return ((v1 * FNV_PRIME) ^ v2) % (0xffffffff);
}

struct CHashimotoResult {
	uint32_t cmix[4];
	uint32_t result[8];
};


static void lyra2re2_hash(const void* input, void* state, int length)
{
	uint32_t hashA[8], hashB[8];

	sph_blake256_context     ctx_blake;
	sph_keccak256_context    ctx_keccak;
	sph_cubehash256_context  ctx_cubehash;
	sph_skein256_context     ctx_skein;
	sph_bmw256_context       ctx_bmw;

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, input, length);
	sph_blake256_close(&ctx_blake, hashA);

	sph_keccak256_init(&ctx_keccak);
	sph_keccak256(&ctx_keccak, hashA, 32);
	sph_keccak256_close(&ctx_keccak, hashB);

	sph_cubehash256_init(&ctx_cubehash);
	sph_cubehash256(&ctx_cubehash, hashB, 32);
	sph_cubehash256_close(&ctx_cubehash, hashA);

	LYRA2(hashB, 32, hashA, 32, hashA, 32, 1, 4, 4);

	sph_skein256_init(&ctx_skein);
	sph_skein256(&ctx_skein, hashB, 32);
	sph_skein256_close(&ctx_skein, hashA);

	sph_cubehash256_init(&ctx_cubehash);
	sph_cubehash256(&ctx_cubehash, hashA, 32);
	sph_cubehash256_close(&ctx_cubehash, hashB);

	sph_bmw256_init(&ctx_bmw);
	sph_bmw256(&ctx_bmw, hashB, 32);
	sph_bmw256_close(&ctx_bmw, hashA);

	memcpy(state, hashA, 32);
}


static struct CHashimotoResult hashimoto(uint8_t *blockToHash, uint32_t *dag, unsigned full_size, int height) {
	uint64_t n = full_size / HASH_BYTES;
	uint64_t mixhashes = MIX_BYTES / HASH_BYTES;
	uint64_t wordhashes = MIX_BYTES / WORD_BYTES;
	uint8_t header[80];
	uint32_t hashedHeader[8];
	memcpy(header, blockToHash, 80);
	lyra2re2_hash((char *)blockToHash, (char*)hashedHeader, 80);
	uint32_t mix[MIX_BYTES / sizeof(uint32_t)];
	for (int i = 0; i < (MIX_BYTES / HASH_BYTES); i++) {
		memcpy(mix + (i * (HASH_BYTES / sizeof(uint32_t))), hashedHeader, HASH_BYTES);
	}
	for (int i = 0; i < ACCESSES; i++) {
		uint32_t p = fnv(i ^ hashedHeader[0], mix[i % (MIX_BYTES / sizeof(uint32_t))]) % (n / mixhashes) * mixhashes;
		uint32_t newdata[MIX_BYTES / sizeof(uint32_t)];
		for (int j = 0; j < mixhashes; j++) {
			uint64_t pj = (p + j) * 8;
			uint32_t* item = dag + pj;
			memcpy(newdata + (j * 8), item, HASH_BYTES);
		}
		for (int i = 0; i < MIX_BYTES / sizeof(uint32_t); i++) {
			mix[i] = fnv(mix[i], newdata[i]);
		}
	}
	uint32_t cmix[4];
	for (int i = 0; i < MIX_BYTES / sizeof(uint32_t); i += 4) {
		cmix[i / 4] = fnv(fnv(fnv(mix[i], mix[i + 1]), mix[i + 2]), mix[i + 3]);
	}
	struct CHashimotoResult result;
	memcpy(result.cmix, cmix, MIX_BYTES / 4);
	uint8_t hash[52];
	memcpy(hash, hashedHeader, 32);
	memcpy(hash + 36, cmix, 16);
	memcpy(hash + 32, &height, 4);
	lyra2re2_hash((char *)hash, (char *)result.result, 52);
	return result;

}

void nightcap_regenhash(struct work *work)
{
	int idx = work->EpochNumber % 2;
	uint32_t *pdata = (uint32_t*)(work->data);
	uint32_t nonce = pdata[19];
	uint32_t endiandata[20];
	unsigned long full_size = nightcap_get_full_size(work->HeightNumber);

	for (int i = 0; i < 20; i++) {
		be32enc(&endiandata[i], pdata[i]);
	}

	applog(LOG_DEBUG, "nightcap_regenhash: nonce check %u for height %u.", nonce, work->HeightNumber);

	cg_rlock(&EthCacheLock[idx]);

	struct CHashimotoResult res = hashimoto((uint8_t*)endiandata, (uint32_t*)(EthCache[idx] + 32), full_size, work->HeightNumber);

	cg_runlock(&EthCacheLock[idx]);

	memcpy(work->hash, res.result, 32);

	char *DbgHash = bin2hex(work->hash, 32);
	applog(LOG_DEBUG, "Regenhash result: %s.", DbgHash);
	free(DbgHash);
}

#if 0

// TOFIX

#define FNV_PRIME		0x01000193

#define fnv(x, y)		(((x) * FNV_PRIME) ^ (y))

#ifdef _MSC_VER
#define restrict __restrict
#endif

typedef struct _DAG64
{
	uint32_t Columns[16];
} DAG64;

uint32_t NightCapCalcEpochNumber(uint8_t *SeedHash)
{
	uint8_t TestSeedHash[32] = { 0 };
	sph_blake256_context ctx_blake;

	for(int Epoch = 0; Epoch < 2048; ++Epoch)
	{
		sph_blake256_init(&ctx_blake);
		sph_blake256(&ctx_blake, TestSeedHash, 32);
		sph_blake256_close(&ctx_blake, TestSeedHash);
		if(!memcmp(TestSeedHash, SeedHash, 32)) return(Epoch + 1);
	}

	applog(LOG_ERR, "Error on epoch calculation.");

	return(0UL);
}

Node CalcDAGItem(const Node *CacheInputNodes, uint32_t NodeCount, uint32_t NodeIdx)
{
	sph_blake256_context ctx_blake;
	Node DAGNode = CacheInputNodes[NodeIdx % NodeCount];

	DAGNode.words[0] ^= NodeIdx;

	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, DAGNode.bytes, 32);
	sph_blake256_close(&ctx_blake, DAGNode.bytes);

	for(uint32_t i = 0; i < 256; ++i)
	{
		uint32_t parent_index = fnv(NodeIdx ^ i, DAGNode.words[i % 8]) % NodeCount;
		Node const *parent = CacheInputNodes + parent_index; //&cache_nodes[parent_index];

		for(int i = 0; i < 16; ++i)
		{
			DAGNode.words[i] *= FNV_PRIME;
			DAGNode.words[i] ^= parent->words[i];
		}
	}
	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, DAGNode.bytes, 32);
	sph_blake256_close(&ctx_blake, DAGNode.bytes);

	return(DAGNode);
}

// OutHash & MixHash MUST have 32 bytes allocated (at least)
void LightNightcap(uint8_t *restrict OutHash, uint8_t *restrict MixHash, const uint8_t *restrict HeaderPoWHash, const Node *Cache, const uint64_t EpochNumber, const uint64_t Nonce)
{
	sph_blake256_context ctx_blake;
	uint32_t MixState[32], TmpBuf[24], NodeCount = NightcapGetCacheSize(EpochNumber) / sizeof(Node);
	uint64_t DagSize;
	Node *EthCache = (Node*) Cache;

	// Initial hash - append nonce to header PoW hash and
	// run it through SHA3 - this becomes the initial value
	// for the mixing state buffer. The init value is used
	// later for the final hash, and is therefore saved.
	memcpy(TmpBuf, HeaderPoWHash, 32UL);
	memcpy(TmpBuf + 8UL, &Nonce, 8UL);
	// TOFIX sha3_512((uint8_t *)TmpBuf, 64UL, (uint8_t *)TmpBuf, 40UL);

	memcpy(MixState, TmpBuf, 64UL);

	// The other half of the state is filled by simply
	// duplicating the first half of its initial value.
	memcpy(MixState + 16UL, MixState, 64UL);

	DagSize = EthGetDAGSize(EpochNumber) / (sizeof(Node) << 1);

	// Main mix of Ethash
	for(uint32_t i = 0, Init0 = MixState[0], MixValue = MixState[0]; i < 64; ++i)
	{
		uint32_t row = fnv(Init0 ^ i, MixValue) % DagSize;
		Node DAGSliceNodes[2];
		DAGSliceNodes[0] = CalcDAGItem(EthCache, NodeCount, row << 1);
		DAGSliceNodes[1] = CalcDAGItem(EthCache, NodeCount, (row << 1) + 1);
		// TOFIX DAG128 *DAGSlice = (DAG128 *)DAGSliceNodes;

		for(uint32_t col = 0; col < 32; ++col)
		{
			// TOFIX MixState[col] = fnv(MixState[col], DAGSlice->Columns[col]);
			MixValue = col == ((i + 1) & 0x1F) ? MixState[col] : MixValue;
		}
	}

	// The reducing of the mix state directly into where
	// it will be hashed to produce the final hash. Note
	// that the initial hash is still in the first 64
	// bytes of TmpBuf - we're appending the mix hash.
	for(int i = 0; i < 8; ++i) TmpBuf[i + 16] = fnv_reduce(MixState + (i << 2));

	memcpy(MixHash, TmpBuf + 16, 32UL);

	// Hash the initial hash and the mix hash concatenated
	// to get the final proof-of-work hash that is our output.
	sha3_256(OutHash, 32UL, (uint8_t *)TmpBuf, 96UL);
}

void ethash_regenhash(struct work *work)
{
	work->Nonce += *((uint32_t *)(work->data + 32));
	applog(LOG_DEBUG, "Regenhash: First qword of input: 0x%016llX.", work->Nonce);
	int idx = work->EpochNumber % 2;
	cg_rlock(&EthCacheLock[idx]);
	LightEthash(work->hash, work->mixhash, work->data, (Node*) (EthCache[idx] + 64), work->EpochNumber, work->Nonce);
	cg_runlock(&EthCacheLock[idx]);

	char *DbgHash = bin2hex(work->hash, 32);

	applog(LOG_DEBUG, "Regenhash result: %s.", DbgHash);
	applog(LOG_DEBUG, "Last ulong: 0x%016llX.", bswap_64(*((uint64_t *)(work->hash + 0))));
	free(DbgHash);
}

#endif
