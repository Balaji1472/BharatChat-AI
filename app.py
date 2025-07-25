
import streamlit as st
import requests
import time
import json
from typing import Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Airavata Chat Assistant",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

FASTAPI_URL = "http://localhost:8000"

def check_api_health() -> bool:
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def send_message_stream(message: str, max_tokens: int = 100, temperature: float = 0.7):
    payload = {
        "message": message,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(
            f"{FASTAPI_URL}/stream", 
            json=payload, 
            stream=True, 
            timeout=90,
            headers={'Accept': 'text/plain'}
        )
        
        if response.status_code == 200:
            return response
        else:
            return None
    except Exception as e:
        st.error(f"Connection error: {str(e)}")
        return None

def send_message(message: str, max_tokens: int = 100, temperature: float = 0.7) -> Dict[Any, Any]:
    try:
        payload = {
            "message": message,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        st.info("Generating response... This may take longer on CPU-only systems. Please wait.")

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
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="TTFT",
            value=f"{metrics_data['ttft_ms']:.1f} ms",
            delta=None
        )
    
    with col2:
        st.metric(
            label="Latency",
            value=f"{metrics_data['latency_ms']:.1f} ms",
            delta=None
        )
    
    with col3:
        st.metric(
            label="Tokens Generated",
            value=metrics_data['tokens_generated'],
            delta=None
        )
    
    with col4:
        st.metric(
            label="Throughput",
            value=f"{metrics_data['tokens_per_second']:.1f} tok/s",
            delta=None
        )

def main():
    st.title("Airavata Chat Assistant")
    st.markdown("*AI4Bharat Quantized Model - GGUF Format*")
    
    st.sidebar.header("Configuration")
    
    api_status = check_api_health()
    if api_status:
        st.sidebar.success("API Connected")
    else:
        st.sidebar.error("API Disconnected")
        st.error("**FastAPI backend is not running!**\n\nPlease start the FastAPI server:\n```bash\npython main.py\n```")
        st.stop()
    
    max_tokens = st.sidebar.slider("Max Tokens", 10, 500, 100)
    temperature = st.sidebar.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
    use_streaming = st.sidebar.checkbox("Enable Streaming", value=True)
    
    st.sidebar.markdown("### Supported Languages")
    st.sidebar.markdown("- English")
    st.sidebar.markdown("- Hindi")
    
    if 'metrics_history' not in st.session_state:
        st.session_state.metrics_history = []
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    st.markdown("### Chat")
    
    for i, (user_msg, bot_msg, metrics) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**You:** {user_msg}")
            st.markdown(f"**Airavata:** {bot_msg}")
            
            with st.expander(f"Performance Metrics (Query {i+1})", expanded=False):
                display_metrics(metrics)
            
            st.markdown("---")
    
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Enter your message:",
            placeholder="Type your message in Hindi or English...\nExample: 'मुझे भारत के बारे में बताओ' or 'What is AI?'",
            height=100
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            submit_button = st.form_submit_button("Send", use_container_width=True)
        with col2:
            clear_button = st.form_submit_button("Clear History", use_container_width=True)
    
    if clear_button:
        st.session_state.chat_history = []
        st.session_state.metrics_history = []
        st.rerun()
    
    if submit_button and user_input.strip():
        if use_streaming:
            with st.spinner("Generating response..."):
                response_stream = send_message_stream(user_input, max_tokens, temperature)
                
                if response_stream:
                    response_container = st.empty()
                    current_response = ""
                    metrics_data = {}
                    
                    try:
                        for line in response_stream.iter_lines(decode_unicode=True):
                            if line.startswith("data: "):
                                data = json.loads(line[6:])
                                
                                if data['type'] == 'token':
                                    current_response += data['text']
                                    response_container.markdown(f"**Airavata:** {current_response}")
                                
                                elif data['type'] == 'ttft':
                                    st.info(f"First token generated in {data['ttft_ms']:.1f}ms")
                                
                                elif data['type'] == 'complete':
                                    metrics_data = {
                                        "latency_ms": data["latency_ms"],
                                        "tokens_generated": data["tokens_generated"],
                                        "tokens_per_second": data["tokens_per_second"],
                                        "ttft_ms": data["ttft_ms"]
                                    }
                                    
                                    st.session_state.chat_history.append((
                                        user_input,
                                        data["response"],
                                        metrics_data
                                    ))
                                    
                                    st.session_state.metrics_history.append({
                                        "query_num": len(st.session_state.metrics_history) + 1,
                                        "latency_ms": data["latency_ms"],
                                        "tokens_per_second": data["tokens_per_second"],
                                        "tokens_generated": data["tokens_generated"],
                                        "ttft_ms": data["ttft_ms"]
                                    })
                                    
                                    break
                                
                                elif data['type'] == 'error':
                                    st.error(f"Error: {data['error']}")
                                    break
                        
                        st.rerun()
                        
                    except Exception as e:
                        st.error(f"Stream processing error: {str(e)}")
                else:
                    st.error("Failed to establish streaming connection")
        else:
            with st.spinner("Generating response..."):
                result = send_message(user_input, max_tokens, temperature)
                
                if result["success"]:
                    response_data = result["data"]
                    
                    st.session_state.chat_history.append((
                        user_input,
                        response_data["response"],
                        {
                            "latency_ms": response_data["latency_ms"],
                            "tokens_generated": response_data["tokens_generated"],
                            "tokens_per_second": response_data["tokens_per_second"],
                            "ttft_ms": response_data["ttft_ms"]
                        }
                    ))
                    
                    st.session_state.metrics_history.append({
                        "query_num": len(st.session_state.metrics_history) + 1,
                        "latency_ms": response_data["latency_ms"],
                        "tokens_per_second": response_data["tokens_per_second"],
                        "tokens_generated": response_data["tokens_generated"],
                        "ttft_ms": response_data["ttft_ms"]
                    })
                    
                    st.rerun()
                else:
                    st.error(f"Error: {result['error']}")
    
    if st.session_state.metrics_history:
        st.markdown("### Performance Analytics")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Latency Over Time', 'Throughput Over Time', 
                          'TTFT Over Time', 'Average Performance'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "indicator"}]]
        )
        
        queries = [m["query_num"] for m in st.session_state.metrics_history]
        latencies = [m["latency_ms"] for m in st.session_state.metrics_history]
        throughputs = [m["tokens_per_second"] for m in st.session_state.metrics_history]
        ttfts = [m["ttft_ms"] for m in st.session_state.metrics_history]
        
        fig.add_trace(
            go.Scatter(x=queries, y=latencies, mode='lines+markers', name='Latency (ms)',
                      line=dict(color='red')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=queries, y=throughputs, mode='lines+markers', name='Throughput (tok/s)',
                      line=dict(color='green')),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Scatter(x=queries, y=ttfts, mode='lines+markers', name='TTFT (ms)',
                      line=dict(color='blue')),
            row=2, col=1
        )
        
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
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Avg Latency", f"{avg_latency:.1f} ms")
        with col2:
            st.metric("Avg Throughput", f"{sum(throughputs)/len(throughputs):.1f} tok/s")
        with col3:
            st.metric("Avg TTFT", f"{sum(ttfts)/len(ttfts):.1f} ms")
        with col4:
            st.metric("Total Queries", len(queries))
        with col5:
            st.metric("Avg Tokens/Query", f"{sum([m['tokens_generated'] for m in st.session_state.metrics_history])/len(st.session_state.metrics_history):.1f}")

    st.markdown("---")
    st.markdown("### Model Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Model**: AI4Bharat/Airavata")
    with col2:
        st.info("**Format**: GGUF (4-bit)")
    with col3:
        st.info("**Size**: ~4GB")

if __name__ == "__main__":
    main()