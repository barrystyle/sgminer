#include <stdint.h>

#include "config.h"
#include "miner.h"
#include "algorithm/ethash.h"

extern cglock_t NightcapCacheLock[2];
extern uint8_t* NightcapCache[2];

#ifdef _MSC_VER
#define restrict __restrict
#endif

typedef struct _DAG64
{
	uint32_t Columns[16];
} DAG64;

typedef union _Node
{
	uint8_t bytes[8 * 4];
	uint32_t words[8];
	uint64_t double_words[8 / 2];
} Node;

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
	sha3_512((uint8_t *)TmpBuf, 64UL, (uint8_t *)TmpBuf, 40UL);

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
		DAG128 *DAGSlice = (DAG128 *)DAGSliceNodes;

		for(uint32_t col = 0; col < 32; ++col)
		{
			MixState[col] = fnv(MixState[col], DAGSlice->Columns[col]);
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
