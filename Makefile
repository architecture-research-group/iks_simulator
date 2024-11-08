
help: 
	@echo "Usage: make [target]"
	@echo "Targets:"
	@echo "  simulator: Compile the simulator"
	@echo "  clean: Remove the simulator"

simulator: main.cpp
	g++ -std=c++17 -O3 -o simulator main.cpp -fopenmp

run_iks_50g_1q: simulator
	@echo "1 IKS, 50GB corpus, 1 query"
	@./simulator 768 68 64 8 1 32552083 1 2>/dev/null

run_iks_50g_16q: simulator
	@echo "1 IKS, 50GB corpus, 16 queries"
	@./simulator 768 68 64 8 1 32552083 16 2>/dev/null

run_iks_50g_64q: simulator
	@echo "1 IKS, 50GB corpus, 64 queries"
	@./simulator 768 68 64 8 1 32552083 64 2>/dev/null

run_iks_200g_1q: simulator
	@echo "1 IKS, 200GB corpus, 1 query"
	@./simulator 768 68 64 8 1 130208332 1 2>/dev/null

run_iks_200g_16q: simulator
	@echo "1 IKS, 200GB corpus, 16 queries"
	@./simulator 768 68 64 8 1 130208332 16 2>/dev/null

run_iks_512g_1q: simulator
	@echo "1 IKS, 512GB corpus, 1 query"
	@./simulator 768 68 64 8 1 333333333 1 2>/dev/null

run_iks_512g_16q: simulator
	@echo "1 IKS, 512GB corpus, 16 queries"
	@./simulator 768 68 64 8 1 333333333 16 2>/dev/null

run_iks_512g_64q: simulator
	@echo "1 IKS, 512GB corpus, 64 queries"
	@./simulator 768 68 64 8 1 333333333 64 2>/dev/null

run_4x_50g_1q: simulator
	@echo "4 IKS, 50GB corpus, 1 query"
	@./simulator 768 68 64 8 4 32552083 1 2>/dev/null

run_4x_50g_16q: simulator
	@echo "4 IKS, 50GB corpus, 16 queries"
	@./simulator 768 68 64 8 4 32552083 16 2>/dev/null

run_4x_200g_1q: simulator
	@echo "4 IKS, 200GB corpus, 1 query"
	@./simulator 768 68 64 8 4 130208332 1 2>/dev/null

run_4x_200g_16q: simulator
	@echo "4 IKS, 200GB corpus, 16 queries"
	@./simulator 768 68 64 8 4 130208332 64 2>/dev/null

run_4x_512g_1q: simulator
	@echo "4 IKS, 512GB corpus, 1 query"
	@./simulator 768 68 64 8 4 333333333 1 2>/dev/null

run_4x_512g_16q: simulator
	@echo "4 IKS, 512GB corpus, 16 queries"
	@./simulator 768 68 64 8 4 333333333 16 2>/dev/null





clean:
	rm -f simulator


