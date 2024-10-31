### Cycle Approximate Simulator for IKS

This is a cycle approximate simulator for IKS. It models the internal structure of IKS, using timings from RTL synthesis as well as directly performing the final stage of aggregation, to approximate the performance of using IKS for exact nearest-neighbor search (ENNS). IKS is composed of 8 near-memory accelerators (NMA) that each have an LPDDR5x module connected over a 128-bit, 8533 MT/s interface. Each NMA is composed of 64 processing engines (PE), each assigned to a query in a batch at runtime (i.e., for batch sizes 64, not all PEs are utilized). Each PE is assigned 68 Float16 multiply-accumulate units that compute dot products between that PE's query vector and a batch of 68 corpus vectors. IKS processes a single dimension of these corpus vectors in each clock cycle, meaning that a D-dimensional vector requires D IKS clock cycles to compute the pairwise distance for a batch of query and corpus vectors. Multiple IKS units may also be used, allowing sharding of an index for strong- or weak-scaling with minimal overhead. Up to 8 IKS units have been tested.

After each batch of corpus vectors, top-k aggregation is done in parallel across all queries in hardware, off the critical path. Between 3 and 5 cycles are required, per corpus vector: 3 cycles are required to check the current top-k list at all, 4 cycles are required to insert a new score and ID without replacement (i.e., very early on, before the entire top-k list is populated), and 5 cycles are required to insert a new score and ID with replacement. These values.

After an offload is complete, each NMA has a partial top-k list for each query vector, which must be aggregated into a single top-k list. This is done on the host, and this simulator gathers timings for this step as well. OpenMP parallelization over NMAs is used to ensure that the core performing final aggregation does not have partial top-k results in its l1 cache. A write to stderr is used to ensure that the main thread (which performs aggregation) cannot get an NMA task. 
 
### Usage

#### Building the simulator:

```bash
make simulator 
```
#### Running the simulator:

```bash
./simulator <vector dimension> <number of mac units> <number of PEs> <number of NMAs> <number of IKS> <corpus size (vectors)> <batch size>
```

#### Generating results:

```bash
make table_3
make table_4
```

#### Examples:

- 1 IKS, 50g corpus, 32 batch size
```bash 
./simulator 768 68 64 8 1 32552084 32 2>/dev/null
```
- Output:
```
Printing to waste time in master thread
Stall cycles: 0
Useful cycles: 45956352
Top-k time: 191 us
Total time: 46.1474 ms
```
