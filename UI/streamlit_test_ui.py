from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# Add parent directory to path to import DataBolt-Edge modules
sys.path.insert(0, str(Path(__file__).parent.parent / "DataBolt-Edge"))

from credentials import load_credentials, validate_credentials
from nvidia_api_management import run_probe


def main():
    st.set_page_config(
        page_title="DataBolt Edge - API Test UI",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Load credentials on app start
    try:
        creds = load_credentials()
    except Exception as e:
        st.error(f"Failed to load credentials: {str(e)}")
        st.stop()

    st.title("⚡ DataBolt Edge - NVIDIA API Test")
    st.markdown("Test your NVIDIA API integration with real-time results")

    # Sidebar config
    with st.sidebar:
        st.header("Configuration")

        # Pre-defined prompts
        prompt_option = st.selectbox(
            "Choose a test prompt:",
            [
                "Custom",
                "What is the capital of France?",
                "Return one short sentence about DataBolt Edge.",
                "Explain quantum computing in simple terms.",
                "What are the top 3 AI trends in 2025?",
            ],
        )

        if prompt_option == "Custom":
            content = st.text_area(
                "Enter your prompt:",
                value="What is the capital of France?",
                height=100,
            )
        else:
            content = prompt_option

        model = st.text_input(
            "Model (optional):",
            value="",
            placeholder="Leave empty for default",
        )

        stream = st.checkbox("Stream response", value=False)

        api_key = st.text_input(
            "API Key (optional):",
            type="password",
            placeholder="Leave empty to use environment variable",
        )

    # Main content area
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Status", "Ready to test")

    with col2:
        st.metric("Model", model or "default")

    with col3:
        st.metric("Streaming", "Yes" if stream else "No")

    st.divider()

    # Test button
    if st.button("🚀 Run Probe Test", use_container_width=True, type="primary"):
        with st.spinner("Running probe..."):
            try:
                # Run the probe
                result = run_probe(
                    content=content,
                    model=model if model else None,
                    stream=stream,
                    api_key=api_key if api_key else None,
                )

                # Display results
                st.success("Probe completed!")

                # Results in columns
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Status",
                        "✅ Success" if result.success else "❌ Failed",
                        delta="OK" if result.success else "Error",
                    )

                with col2:
                    st.metric("Latency", f"{result.latency_ms:.2f}ms")

                with col3:
                    if result.status_code is not None:
                        st.metric("HTTP Status", result.status_code)
                    else:
                        st.metric("HTTP Status", "N/A")

                st.divider()

                # Detailed results
                st.subheader("Details")

                if result.success and result.response:
                    # Extract response content
                    choices = result.response.get("choices", [])
                    if isinstance(choices, list) and choices:
                        message = choices[0].get("message", {})
                        response_content = message.get("content", "No content")

                        st.write("**Response:**")
                        st.info(response_content)

                    # Show full response JSON
                    with st.expander("📋 Full Response (JSON)"):
                        st.json(result.response)

                elif result.error:
                    st.error(f"**Error:** {result.error}")

                # Test history
                st.divider()
                st.subheader("Test Summary")

                summary_col1, summary_col2 = st.columns(2)
                with summary_col1:
                    st.write("**Prompt:**")
                    st.code(content, language="text")

                with summary_col2:
                    st.write("**Metadata:**")
                    metadata = {
                        "Model": model or "default",
                        "Stream": stream,
                        "Latency (ms)": f"{result.latency_ms:.2f}",
                        "Success": result.success,
                    }
                    st.table(metadata.items())

            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                st.exception(e)

    st.divider()

    # Footer with instructions
    with st.expander("ℹ️ Instructions"):
        st.markdown(
            """
        1. **Configure** your test parameters in the sidebar
        2. **Choose** a predefined prompt or enter a custom one
        3. **Click** the "Run Probe Test" button
        4. **View** results including latency and response

        **Environment Setup:**
        - Ensure `NVIDIA_BEARER_TOKEN` is set in your environment
        - Or provide an API key in the sidebar (not recommended for production)

        **Tip:** Use the expander to view the full JSON response for debugging
        """
        )


if __name__ == "__main__":
    main()
