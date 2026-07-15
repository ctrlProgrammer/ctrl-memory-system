#!/usr/bin/env bash
# ── ctrl-memory installer ──────────────────────────────────────────────
# Installs ctrl-memory in an isolated virtual environment and makes the
# `ctrl-memory-mcp` command available globally.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/ctrlProgrammer/ctrl-memory-system/main/install.sh | bash
#   # or
#   ./install.sh
#
# What it does:
#   1. Creates a venv at ~/.local/share/ctrl-memory/venv
#   2. Installs ctrl-memory and optional dependencies
#   3. Symlinks ctrl-memory-mcp -> ~/.local/bin/ctrl-memory-mcp
#   4. Optionally configures Hermes Agent (auto-detects if Hermes is installed)
# ───────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="ctrlProgrammer/ctrl-memory-system"
APP_NAME="ctrl-memory"
VENV_DIR="$HOME/.local/share/$APP_NAME/venv"
BIN_DIR="$HOME/.local/bin"
HERMES_PLUGIN_DIR="$HOME/.hermes/hermes-agent/plugins/memory/$APP_NAME"
PIP_INSTALL_FLAGS="--quiet"

# ── Colors ────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}ℹ${NC}  $1"; }
ok()    { echo -e "${GREEN}✔${NC}  $1"; }
warn()  { echo -e "${YELLOW}⚠${NC}  $1"; }
err()   { echo -e "${RED}✘${NC}  $1"; }

# ── Pre-flight checks ─────────────────────────────────────────────────

# Check for Python 3.11+
PYTHON=""
for cmd in python3.12 python3.11 python3; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+')
        major="${ver%.*}"
        minor="${ver#*.}"
        if [ "$major" -ge 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    err "Python 3.11+ is required but not found."
    err "Install it with your package manager:"
    err "  sudo apt install python3.12 python3.12-venv  # Debian/Ubuntu"
    err "  sudo dnf install python3.12                   # Fedora"
    err "  brew install python@3.12                      # macOS"
    exit 1
fi
ok "Found $("$PYTHON" --version) at $(command -v "$PYTHON")"

# Check for pip
if ! "$PYTHON" -m pip --version &>/dev/null; then
    err "pip is not available for $PYTHON."
    err "Install it: $PYTHON -m ensurepip --upgrade"
    exit 1
fi

# Ensure BIN_DIR is in PATH
mkdir -p "$BIN_DIR"
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    warn "$BIN_DIR is not in your PATH."
    warn "Add this to your ~/.bashrc or ~/.zshrc:"
    echo "    export PATH=\"\$PATH:$BIN_DIR\""
fi

# ── Install ───────────────────────────────────────────────────────────

echo ""
info "Installing $APP_NAME into isolated venv..."
echo "    Venv: $VENV_DIR"
echo "    Link: $BIN_DIR/$APP_NAME-mcp"
echo ""

# Create venv if needed
if [ ! -f "$VENV_DIR/bin/$PYTHON" ]; then
    info "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created."
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Upgrade pip itself in the venv
pip install --quiet --upgrade pip 2>&1 | tail -1 || true

# Determine install mode
if [ -d "$(dirname "$0")/memory_backend.py" ] || [ -f "$(dirname "$0")/pyproject.toml" ]; then
    # Local install (running from project directory)
    INSTALL_SRC="$(dirname "$0")"
    info "Installing from local source: $INSTALL_SRC"
    pip install $PIP_INSTALL_FLAGS -e "$INSTALL_SRC" 2>&1 | tail -1
else
    # Remote install from GitHub
    info "Installing from GitHub..."
    pip install $PIP_INSTALL_FLAGS "git+https://github.com/$REPO.git" 2>&1 | tail -1
fi

# Optional: try installing sentence-transformers for embeddings
if pip install $PIP_INSTALL_FLAGS "sentence-transformers>=3.0.0" 2>&1 | tail -1; then
    ok "Semantic search enabled (sentence-transformers installed)."
else
    warn "Semantic search not available (sentence-transformers install failed)."
    warn "Install manually later: pip install 'ctrl-memory[embeddings]'"
fi

# Symlink the launcher
mkdir -p "$BIN_DIR"
LAUNCHER="$BIN_DIR/$APP_NAME-mcp"
cat > "$LAUNCHER" << LAUNCHER_EOF
#!/usr/bin/env bash
exec "$VENV_DIR/bin/$APP_NAME-mcp" "\$@"
LAUNCHER_EOF
chmod +x "$LAUNCHER"

# Verify
if command -v "$APP_NAME-mcp" &>/dev/null; then
    ok "Command '$APP_NAME-mcp' is available."
else
    warn "Command not found in current shell. Run: exec \$SHELL"
fi

# ── Optional: Hermes Agent configuration ──────────────────────────────

if [ -d "$HOME/.hermes" ]; then
    echo ""
    info "Hermes Agent detected. Configuring ctrl-memory plugin..."

    # Copy plugin files
    if [ -d "$(dirname "$0")/hermes_provider" ]; then
        PLUGIN_SRC="$(dirname "$0")/hermes_provider"
    else
        # Download from GitHub
        PLUGIN_SRC=$(mktemp -d)
        curl -fsSL "https://api.github.com/repos/$REPO/contents/hermes_provider" \
          | grep '"download_url"' | awk '{print $2}' | tr -d '",' \
          | while read -r url; do
                curl -fsSL "$url" -o "$PLUGIN_SRC/$(basename $url)"
            done 2>/dev/null || true
    fi

    mkdir -p "$HERMES_PLUGIN_DIR"
    cp -r "$PLUGIN_SRC/"* "$HERMES_PLUGIN_DIR/" 2>/dev/null || true
    ok "Plugin files copied to $HERMES_PLUGIN_DIR"

    # Update Hermes config
    HERMES_CONFIG="$HOME/.hermes/config.yaml"
    if [ -f "$HERMES_CONFIG" ]; then
        # Check if provider is already set
        if grep -q "provider: ctrl-memory" "$HERMES_CONFIG" 2>/dev/null; then
            ok "Hermes already configured to use ctrl-memory."
        else
            warn "Run these commands to activate in Hermes:"
            echo ""
            echo "    hermes config set memory.memory_enabled true"
            echo "    hermes config set memory.provider ctrl-memory"
            echo ""
        fi
    fi
else
    echo ""
    info "Hermes Agent not detected. Skipping Hermes configuration."
fi

# ── Summary ───────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  $APP_NAME installed successfully!${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Available commands:"
echo "    $APP_NAME-mcp          Start the MCP stdio server"
echo ""
echo "  Quick test:"
echo "    echo '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"initialize\",\"params\":{\"protocolVersion\":\"2024-11-05\",\"capabilities\":{}}}' | $APP_NAME-mcp"
echo ""
echo "  Add a fact:"
echo '    echo '\''{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"add_memory","arguments":{"user_id":"default","content":"Remember: I prefer Fastify over Express"}}}'\'' | '"$APP_NAME-mcp"
echo ""
echo "  Uninstall:"
echo "    rm -rf $VENV_DIR $LAUNCHER"
echo ""