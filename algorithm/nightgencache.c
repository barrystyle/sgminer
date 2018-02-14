#include <stdint.h>

#include "sph/sph_blake.h"

typedef union node
{
	uint8_t bytes[8 * 4];
	uint32_t words[8];
	uint64_t double_words[8 / 2];
} node;

// Output (cache_nodes) MUST have at least cache_size bytes
void NightcapGenerateCache(void *cache_nodes_in, uint8_t* const seedhash, uint64_t cache_size)
{
	sph_blake256_context ctx_blake;
	uint32_t const num_nodes = (uint32_t)(cache_size / sizeof(node));
	node *cache_nodes = (node *)cache_nodes_in;
	sph_blake256_init(&ctx_blake);
	sph_blake256(&ctx_blake, seedhash, 32);
	sph_blake256_close(&ctx_blake, cache_nodes[0].bytes);

	for(uint32_t i = 1; i != num_nodes; ++i)
	{
		sph_blake256_init(&ctx_blake);
		sph_blake256(&ctx_blake, cache_nodes[i - 1].bytes, 32);
		sph_blake256_close(&ctx_blake, cache_nodes[i].bytes);
	}

	for(uint32_t j = 0; j < 3; j++) // this one can be unrolled entirely, ETHASH_CACHE_ROUNDS is constant
	{
		for(uint32_t i = 0; i != num_nodes; i++)
		{
			uint32_t const idx = cache_nodes[i].words[0] % num_nodes;
			node data;
			data = cache_nodes[(num_nodes - 1 + i) % num_nodes];
			for(uint32_t w = 0; w != 16; ++w) // this one can be unrolled entirely as well
			{
				data.words[w] ^= cache_nodes[idx].words[w];
			}
			sph_blake256_init(&ctx_blake);
			sph_blake256(&ctx_blake, data.bytes, 32);
			sph_blake256_close(&ctx_blake, cache_nodes[i].bytes);
		}
	}
}
