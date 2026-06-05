#!/bin/bash
set -e

# Clone
git clone https://github.com/winsinor/haiku-bot.git
cd haiku-bot

# Python deps
pip install requests syllables pronouncing

# Pull the model (requires Ollama to be running)
ollama pull gemma2:2b

echo ""
echo "Ready. Run with:"
echo "  python3 haiku_bot.py"
