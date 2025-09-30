#!/bin/bash
# TIER 1 Rejuvenation Suite - Environment Management Script
# ========================================================
#
# This script helps manage conda environments and dependency locking
# for the TIER 1 Cellular Rejuvenation Suite.
#
# Usage:
#   ./env_manager.sh create-dev     # Create development environment
#   ./env_manager.sh create-prod    # Create production environment  
#   ./env_manager.sh update-lock    # Update lock files
#   ./env_manager.sh install        # Install package in current env
#   ./env_manager.sh test           # Test installation

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Check if conda is available
check_conda() {
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed or not in PATH"
        print_info "Please install Miniconda or Anaconda first"
        exit 1
    fi
    print_info "Conda found: $(conda --version)"
}

# Create development environment
create_dev_env() {
    print_info "Creating development environment..."
    
    if conda env list | grep -q "tier1-rejuvenation-suite"; then
        print_warning "Environment 'tier1-rejuvenation-suite' already exists"
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n tier1-rejuvenation-suite
        else
            print_info "Using existing environment"
            return
        fi
    fi
    
    conda env create -f "${SCRIPT_DIR}/environment.yml"
    print_success "Development environment created successfully"
}

# Create production environment  
create_prod_env() {
    print_info "Creating production environment..."
    
    if conda env list | grep -q "tier1-rejuvenation-suite-prod"; then
        print_warning "Environment 'tier1-rejuvenation-suite-prod' already exists"
        read -p "Remove and recreate? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            conda env remove -n tier1-rejuvenation-suite-prod
        else
            print_info "Using existing environment"
            return
        fi
    fi
    
    conda env create -f "${SCRIPT_DIR}/environment-prod.yml"
    print_success "Production environment created successfully"
}

# Update dependency lock files
update_lock_files() {
    print_info "Updating dependency lock files..."
    
    # Export current environment to lock file
    if conda env list | grep -q "tier1-rejuvenation-suite"; then
        print_info "Exporting development environment..."
        conda env export -n tier1-rejuvenation-suite > "${SCRIPT_DIR}/environment-lock.yml"
        print_success "Development lock file updated: environment-lock.yml"
    fi
    
    if conda env list | grep -q "tier1-rejuvenation-suite-prod"; then
        print_info "Exporting production environment..."
        conda env export -n tier1-rejuvenation-suite-prod > "${SCRIPT_DIR}/environment-prod-lock.yml"
        print_success "Production lock file updated: environment-prod-lock.yml"
    fi
    
    # Update pip requirements
    if [[ -f "${SCRIPT_DIR}/requirements.in" ]]; then
        print_info "Compiling pip requirements..."
        if command -v pip-compile &> /dev/null; then
            pip-compile "${SCRIPT_DIR}/requirements.in" --output-file "${SCRIPT_DIR}/requirements-lock.txt"
            print_success "Pip requirements compiled: requirements-lock.txt"
        else
            print_warning "pip-tools not found, skipping pip requirements compilation"
        fi
    fi
}

# Install package in current environment
install_package() {
    print_info "Installing TIER 1 Rejuvenation Suite in current environment..."
    
    cd "${SCRIPT_DIR}"
    
    # Install in development mode
    pip install -e .
    
    print_success "Package installed successfully"
    print_info "Test with: tier1 --help"
}

# Test installation
test_installation() {
    print_info "Testing TIER 1 Rejuvenation Suite installation..."
    
    cd "${SCRIPT_DIR}"
    
    if [[ -f "test_installation.py" ]]; then
        python test_installation.py
    else
        print_warning "test_installation.py not found, running basic tests..."
        
        # Basic CLI tests
        tier1 --version
        tier1 info
        tier1 bulk --help
        tier1 sc --help  
        tier1 multi --help
        
        print_success "Basic tests passed"
    fi
}

# Export current environment specifications  
export_specs() {
    print_info "Exporting current environment specifications..."
    
    # Get current conda environment name
    ENV_NAME="${CONDA_DEFAULT_ENV:-base}"
    
    if [[ "$ENV_NAME" == "base" ]]; then
        print_warning "Currently in base environment. Activate target environment first."
        return 1
    fi
    
    print_info "Exporting environment: $ENV_NAME"
    
    # Export with exact versions
    conda env export > "${SCRIPT_DIR}/environment-${ENV_NAME}-$(date +%Y%m%d).yml"
    
    # Export pip freeze  
    pip freeze > "${SCRIPT_DIR}/requirements-${ENV_NAME}-$(date +%Y%m%d).txt"
    
    print_success "Environment specifications exported"
}

# Build Docker image
build_docker() {
    print_info "Building Docker image..."
    
    cd "${SCRIPT_DIR}"
    
    # Build production image
    docker build -t tier1-rejuvenation-suite:latest --target production .
    
    # Build development image
    docker build -t tier1-rejuvenation-suite:dev --target development .
    
    print_success "Docker images built successfully"
    print_info "Run with: docker run -it tier1-rejuvenation-suite:latest tier1 --help"
}

# Clean up environments and caches
cleanup() {
    print_info "Cleaning up environments and caches..."
    
    # Clean conda cache
    conda clean --all -y
    
    # Clean pip cache
    pip cache purge
    
    # Remove lock files older than 30 days
    find "${SCRIPT_DIR}" -name "*-lock*.yml" -mtime +30 -delete 2>/dev/null || true
    find "${SCRIPT_DIR}" -name "*-lock*.txt" -mtime +30 -delete 2>/dev/null || true
    
    print_success "Cleanup completed"
}

# Show help
show_help() {
    cat << EOF
TIER 1 Rejuvenation Suite - Environment Manager

Usage: $0 <command>

Commands:
  create-dev      Create development environment from environment.yml
  create-prod     Create production environment from environment-prod.yml  
  update-lock     Update all dependency lock files
  install         Install package in current environment
  test           Test installation with validation script
  export-specs    Export current environment specifications
  build-docker    Build Docker images for production and development
  cleanup         Clean up caches and old lock files
  help           Show this help message

Examples:
  $0 create-dev                    # Set up development environment
  $0 install                       # Install package in current env
  $0 test                         # Validate installation
  $0 update-lock                  # Update lock files
  conda activate tier1-rejuvenation-suite && $0 install

Environment Files:
  environment.yml           - Full development environment
  environment-prod.yml      - Minimal production environment  
  requirements.in           - Pip requirements specification
  requirements.txt          - Pinned pip requirements
  Dockerfile               - Multi-stage Docker build
  docker-compose.yml       - Docker services configuration

For more information, see README.md
EOF
}

# Main script logic
main() {
    case "${1:-help}" in
        "create-dev")
            check_conda
            create_dev_env
            ;;
        "create-prod")
            check_conda  
            create_prod_env
            ;;
        "update-lock")
            check_conda
            update_lock_files
            ;;
        "install")
            install_package
            ;;
        "test")
            test_installation
            ;;
        "export-specs")
            check_conda
            export_specs
            ;;
        "build-docker")
            build_docker
            ;;
        "cleanup")
            check_conda
            cleanup
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            print_error "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"