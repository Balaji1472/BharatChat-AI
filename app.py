import streamlit as st
import requests
import time
import json
from typing import Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page config
st.set_page_config(
    page_title="Airavata Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
FASTAPI_URL = "http://localhost:8000"

def check_api_health() -> bool:
    """Check if FastAPI backend is running"""
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def send_message(message: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[Any, Any]:
    """Send message to FastAPI backend"""
    try:
        payload = {
            "message": message,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        # Show warning to user for longer response time
        st.info("⏳ Generating response... This may take longer on CPU-only systems. Please wait.")

        response = requests.post(f"{FASTAPI_URL}/generate", json=payload, timeout=90)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"API Error: {response.status_code}"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timeout (90s)"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def display_metrics(metrics_data):
    """Display performance metrics"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Latency",
            value=f"{metrics_data['latency_ms']:.1f} ms",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Tokens Generated",
            value=metrics_data['tokens_generated'],
            delta=None
        )
    
    with col3:
        st.metric(
            label="Throughput",
            value=f"{metrics_data['tokens_per_second']:.1f} tok/s",
            delta=None
        )

def main():
    # Title and header
    st.title("🤖 Airavata Chat Assistant")
    st.markdown("*AI4Bharat Quantized Model - GGUF Format*")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # API Health Check
    api_status = check_api_health()
    if api_status:
        st.sidebar.success("✅ API Connected")
    else:
        st.sidebar.error("❌ API Disconnected")
        st.error("🚨 **FastAPI backend is not running!**\n\nPlease start the FastAPI server:\n```bash\npython main.py\n```")
        st.stop()
    
    # Model parameters
    max_tokens = st.sidebar.slider("Max Tokens", 10, 500, 100)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    
    # Language info
    st.sidebar.markdown("### 🌐 Supported Languages")
    st.sidebar.markdown("- English")
    st.sidebar.markdown("- हिंदी (Hindi)")
    
    # Performance tracking
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    
    # Chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Main chat interface
    st.markdown("### 💬 Chat")
    
    # Display chat history
    for i, (user_msg, bot_msg, metrics) in enumerate(st.session_state.chat_history):
        with st.container():
            # User message
            st.markdown(f"**🧑‍💻 You:** {user_msg}")
            
            # Bot response
            st.markdown(f"**🤖 Airavata:** {bot_msg}")
            
            # Metrics (collapsible)
            with st.expander(f"📊 Performance Metrics (Query {i+1})", expanded=False):
                display_metrics(metrics)
            
            st.markdown("---")
    
    # Input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Enter your message:",
            placeholder="Type your message in Hindi or English...\nExample: 'मुझे भारत के बारे में बताओ' or 'What is AI?'",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.form_submit_button("Send 📤", use_container_width=True)
        with col2:
            clear_button = st.form_submit_button("Clear History 🗑️", use_container_width=True)
    
    # Clear history
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.metrics_history = []
        st.rerun()
    
    # Process user input
    if submit_button and user_input.strip():
        with st.spinner("🔄 Generating response..."):
            result = send_message(user_input, max_tokens, temperature)
            
            if result["success"]:
                response_data = result["data"]
                
                # Add to chat history
                st.session_state.chat_history.append((
                    user_input,
                    response_data["response"],
                    {
                        "latency_ms": response_data["latency_ms"],
                        "tokens_generated": response_data["tokens_generated"],
                        "tokens_per_second": response_data["tokens_per_second"]
                    }
                ))
                
                # Add to metrics history
                st.session_state.metrics_history.append({
                    "query_num": len(st.session_state.metrics_history) + 1,
                    "latency_ms": response_data["latency_ms"],
                    "tokens_per_second": response_data["tokens_per_second"],
                    "tokens_generated": response_data["tokens_generated"]
                })
                
                st.rerun()
            else:
                st.error(f"❌ Error: {result['error']}")
    
    # Performance Analytics Section
    if st.session_state.metrics_history:
        st.markdown("### 📈 Performance Analytics")
        
        # Create performance charts
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Latency Over Time', 'Throughput Over Time', 
                          'Tokens Generated', 'Average Performance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "indicator"}]]
        )
        
        queries = [m["query_num"] for m in st.session_state.metrics_history]
        latencies = [m["latency_ms"] for m in st.session_state.metrics_history]
        throughputs = [m["tokens_per_second"] for m in st.session_state.metrics_history]
        tokens = [m["tokens_generated"] for m in st.session_state.metrics_history]
        
        # Latency chart
        fig.add_trace(
            go.Scatter(x=queries, y=latencies, mode='lines+markers', name='Latency (ms)',
                      line=dict(color='red')),
            row=1, col=1
        )
        
        # Throughput chart
        fig.add_trace(
            go.Scatter(x=queries, y=throughputs, mode='lines+markers', name='Throughput (tok/s)',
                      line=dict(color='green')),
            row=1, col=2
        )
        
        # Tokens generated chart
        fig.add_trace(
            go.Bar(x=queries, y=tokens, name='Tokens Generated',
                   marker_color='blue'),
            row=2, col=1
        )
        
        # Average performance indicator
        avg_latency = sum(latencies) / len(latencies)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=avg_latency,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Avg Latency (ms)"},
                gauge={'axis': {'range': [None, max(latencies) * 1.2]},
                       'bar': {'color': "darkblue"},
                       'steps': [{'range': [0, avg_latency], 'color': "lightgray"}],
                       'threshold': {'line': {'color': "red", 'width': 4},
                                   'thickness': 0.75, 'value': avg_latency}}
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=600, showlegend=False, title_text="Model Performance Dashboard")
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Latency", f"{avg_latency:.1f} ms")
        with col2:
            st.metric("Avg Throughput", f"{sum(throughputs)/len(throughputs):.1f} tok/s")
        with col3:
            st.metric("Total Queries", len(queries))
        with col4:
            st.metric("Avg Tokens/Query", f"{sum(tokens)/len(tokens):.1f}")

    # Footer
    st.markdown("---")
    st.markdown("### 📋 Model Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Model**: AI4Bharat/Airavata")
    with col2:
        st.info("**Format**: GGUF (4-bit)")
    with col3:
        st.info("**Size**: ~4GB")

if __name__ == "__main__":
    main()