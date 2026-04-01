.PHONY: build up down logs health test benchmark clean

# Build all service images
build:
	docker compose build

# Build a single service
build-%:
	docker compose build $*

# Start all services
up:
	docker compose up -d

# Start specific service(s)
up-%:
	docker compose up -d $*

# Stop all services
down:
	docker compose down

# View logs (all or specific service)
logs:
	docker compose logs -f

logs-%:
	docker compose logs -f $*

# Check health of all services
health:
	@echo "=== Service Health ==="
	@curl -s http://localhost:8080/healthz | python3 -m json.tool 2>/dev/null || echo "Gateway: DOWN"
	@echo ""
	@curl -s http://localhost:8081/healthz | python3 -m json.tool 2>/dev/null || echo "Paddle: DOWN"
	@echo ""
	@curl -s http://localhost:8082/healthz | python3 -m json.tool 2>/dev/null || echo "Tesseract: DOWN"
	@echo ""
	@curl -s http://localhost:8083/healthz | python3 -m json.tool 2>/dev/null || echo "Surya: DOWN"
	@echo ""
	@curl -s http://localhost:8084/healthz | python3 -m json.tool 2>/dev/null || echo "docTR: DOWN"

# Run benchmark
benchmark:
	python3 benchmark/run.py --host $$(cat infra/.deploy-state) --fixtures-dir test-fixtures/

# Run resource consumption benchmark
benchmark-resources:
	python3 benchmark/resource_bench.py --host $$(cat infra/.deploy-state)

# Run only throughput phase
benchmark-throughput:
	python3 benchmark/resource_bench.py --host $$(cat infra/.deploy-state) --phase throughput

# Quick test: send a file to all services via gateway
test-compare:
	@if [ -z "$(FILE)" ]; then echo "Usage: make test-compare FILE=path/to/image.png"; exit 1; fi
	curl -s -F "file=@$(FILE)" http://localhost:8080/api/v1/compare | python3 -m json.tool

# Test individual service
test-paddle:
	@if [ -z "$(FILE)" ]; then echo "Usage: make test-paddle FILE=path/to/image.png"; exit 1; fi
	curl -s -F "file=@$(FILE)" http://localhost:8081/api/v1/ocr | python3 -m json.tool

test-tesseract:
	@if [ -z "$(FILE)" ]; then echo "Usage: make test-tesseract FILE=path/to/image.png"; exit 1; fi
	curl -s -F "file=@$(FILE)" http://localhost:8082/api/v1/ocr | python3 -m json.tool

test-surya:
	@if [ -z "$(FILE)" ]; then echo "Usage: make test-surya FILE=path/to/image.png"; exit 1; fi
	curl -s -F "file=@$(FILE)" http://localhost:8083/api/v1/ocr | python3 -m json.tool

test-doctr:
	@if [ -z "$(FILE)" ]; then echo "Usage: make test-doctr FILE=path/to/image.png"; exit 1; fi
	curl -s -F "file=@$(FILE)" http://localhost:8084/api/v1/ocr | python3 -m json.tool

# Remove all containers, volumes, and images
clean:
	docker compose down -v --rmi local
