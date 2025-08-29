import streamlit as st
from pyngrok import ngrok
import subprocess
import sys

# Set the port for Streamlit
port = 8501

# Start ngrok tunnel
public_url = ngrok.connect(port)
print(f"Ngrok tunnel URL: {public_url}")

# Launch Streamlit app
subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py", f"--server.port={port}"])

