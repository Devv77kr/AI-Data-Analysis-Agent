import tempfile
import csv
import logging
import os
from datetime import datetime
import streamlit as st
import pandas as pd
from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.duckdb import DuckDbTools
from agno.tools.pandas import PandasTools

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="📊 Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
def initialize_session_state():
    """Initialize all session state variables"""
    if "openai_key" not in st.session_state:
        st.session_state.openai_key = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "dataframe" not in st.session_state:
        st.session_state.dataframe = None
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "duckdb_tools" not in st.session_state:
        st.session_state.duckdb_tools = None

initialize_session_state()

# Function to validate dataframe
def validate_dataframe(df):
    """Validate and provide insights about uploaded data"""
    if df.empty:
        st.error("❌ Uploaded file is empty")
        return False
    
    if len(df) > 100000:
        st.warning(f"⚠️ Large dataset detected ({len(df):,} rows). Processing may be slow.")
    
    return True

# Function to display data insights
def display_data_insights(df):
    """Display comprehensive data insights"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        null_percentage = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Missing Data %", f"{null_percentage:.2f}%")
    
    # Data types summary
    with st.expander("📋 Data Types Summary"):
        dtype_summary = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values,
            'Null Count': df.isnull().sum().values
        })
        st.dataframe(dtype_summary, use_container_width=True)

# Function to preprocess and save the uploaded file
@st.cache_data
def preprocess_and_save(file_name, file_content):
    """Preprocess and save the uploaded file with enhanced error handling"""
    try:
        logger.info(f"Processing file: {file_name}")
        
        # Read the uploaded file into a DataFrame
        if file_name.endswith('.csv'):
            df = pd.read_csv(
                file_content,
                encoding='utf-8',
                na_values=['NA', 'N/A', 'missing', 'None', 'null', '']
            )
        elif file_name.endswith('.xlsx'):
            df = pd.read_excel(
                file_content,
                na_values=['NA', 'N/A', 'missing', 'None', 'null', '']
            )
        else:
            st.error("❌ Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None
        
        # Validate dataframe
        if not validate_dataframe(df):
            return None, None, None
        
        # Smart type conversion
        df = smart_type_conversion(df)
        
        # Ensure string columns are properly quoted
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)
        
        # Create a temporary file to save the preprocessed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)
        
        logger.info(f"File processed successfully. Temp path: {temp_path}")
        return temp_path, df.columns.tolist(), df
        
    except UnicodeDecodeError as e:
        st.error("❌ File encoding issue. Try UTF-8 or Latin-1 encoding.")
        logger.error(f"Encoding error: {e}")
        return None, None, None
    except pd.errors.EmptyDataError:
        st.error("❌ File is empty.")
        logger.error("Empty data error")
        return None, None, None
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        logger.error(f"File processing error: {e}")
        return None, None, None

# Function for smart type conversion
def smart_type_conversion(df):
    """Intelligent type conversion for better data analysis"""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Try numeric conversion first
            try:
                numeric_df = pd.to_numeric(df[col], errors='coerce')
                if numeric_df.notna().sum() / len(df) > 0.8:  # If 80% can be converted
                    df[col] = numeric_df
                    continue
            except (ValueError, TypeError):
                pass
            
            # Try datetime conversion if numeric failed
            try:
                datetime_df = pd.to_datetime(df[col], errors='coerce')
                if datetime_df.notna().sum() / len(df) > 0.8:
                    df[col] = datetime_df
                    continue
            except (ValueError, TypeError):
                pass
    
    return df

# Function to initialize agent
@st.cache_resource
def initialize_agent(api_key):
    """Initialize the data analyst agent with caching"""
    try:
        logger.info("Initializing data analyst agent")
        
        duckdb_tools = DuckDbTools()
        
        agent = Agent(
            model=OpenAIChat(id="gpt-4o", api_key=api_key),
            tools=[duckdb_tools, PandasTools()],
            system_message="""You are an expert data analyst with deep knowledge of SQL, statistics, and data visualization. 
Your responsibilities:
1. Use the 'uploaded_data' table to answer user queries accurately
2. Generate efficient SQL queries using DuckDB to solve problems
3. Provide clear, concise, and actionable insights
4. Include relevant statistics and findings in your analysis
5. Suggest follow-up analyses when appropriate
6. Always explain your methodology and findings in simple terms
When providing code, format it clearly with proper syntax highlighting.
Be proactive in identifying data quality issues and patterns.""",
            markdown=True,
        )
        
        st.session_state.duckdb_tools = duckdb_tools
        logger.info("Agent initialized successfully")
        return agent
        
    except Exception as e:
        logger.error(f"Agent initialization failed: {e}")
        st.error(f"❌ Failed to initialize agent: {str(e)}")
        return None

# Streamlit app
st.title("📊 Data Analyst Agent")
st.markdown("Analyze your data using AI-powered natural language queries powered by GPT-4o")

# Sidebar for API keys and settings
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    openai_key = st.text_input(
        "Enter your OpenAI API key:",
        type="password",
        help="Your API key is not stored and only used for this session"
    )
    
    if openai_key:
        st.session_state.openai_key = openai_key
        st.success("✅ API key saved!")
    else:
        st.warning("⚠️ Please enter your OpenAI API key to proceed.")
    
    # Display query history
    st.divider()
    st.subheader("📜 Query History")
    if st.session_state.query_history:
        for i, query in enumerate(st.session_state.query_history[-5:], 1):
            st.caption(f"{i}. {query['query'][:50]}...")
        
        if st.button("🗑️ Clear History"):
            st.session_state.query_history = []
            st.rerun()
    else:
        st.caption("No queries yet")

# Main content area
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📁 Upload Data")

with col2:
    if st.button("🔄 Reset Analysis"):
        st.session_state.dataframe = None
        st.session_state.agent = None
        st.session_state.query_history = []
        st.rerun()

# File upload widget
uploaded_file = st.file_uploader(
    "Upload a CSV or Excel file",
    type=["csv", "xlsx"],
    help="Supported formats: CSV (.csv), Excel (.xlsx)"
)

if uploaded_file is not None and st.session_state.openai_key:
    # Preprocess and save the uploaded file
    temp_path, columns, df = preprocess_and_save(uploaded_file.name, uploaded_file)
    
    if temp_path and columns and df is not None:
        st.session_state.dataframe = df
        
        # Display data insights
        display_data_insights(df)
        
        # Display the uploaded data as an interactive table
        with st.expander("📊 View Full Dataset", expanded=False):
            st.dataframe(df, use_container_width=True)
        
        # Initialize agent if not already done
        if st.session_state.agent is None:
            st.session_state.agent = initialize_agent(st.session_state.openai_key)
        
        if st.session_state.agent and st.session_state.duckdb_tools:
            try:
                # Load the CSV file into DuckDB
                st.session_state.duckdb_tools.load_local_csv_to_table(
                    path=temp_path,
                    table="uploaded_data",
                )
            except Exception as e:
                logger.error(f"DuckDB loading error: {e}")
                st.error(f"❌ Error loading data to DuckDB: {str(e)}")
        
        st.divider()
        
        # Query interface
        st.subheader("🔍 Query Your Data")
        st.caption("Ask questions about your data in natural language")
        
        # Query input with examples
        col1, col2 = st.columns([4, 1])
        with col1:
            user_query = st.text_area(
                "Ask a query about the data:",
                placeholder="e.g., 'What are the top 5 most common values in the status column?'",
                height=80
            )
        with col2:
            st.caption("**Example queries:**")
            st.caption("- Summary stats")
            st.caption("- Top values")
            st.caption("- Trends")
            st.caption("- Correlations")
        
        # Info message about processing
        st.info("💡 Agent is analyzing your data. Results will appear below.")
        
        if st.button("▶️ Submit Query", use_container_width=True):
            if not user_query.strip():
                st.warning("⚠️ Please enter a query.")
            elif not st.session_state.agent:
                st.error("❌ Agent not initialized. Please check your API key.")
            else:
                try:
                    # Show loading spinner while processing
                    with st.spinner('🔄 Processing your query...'):
                        # Log the query
                        logger.info(f"Processing query: {user_query}")
                        
                        # Get the response from the agent
                        response = st.session_state.agent.run(user_query)
                        
                        # Extract the content from the response object
                        if hasattr(response, 'content'):
                            response_content = response.content
                        else:
                            response_content = str(response)
                        
                        # Add to history
                        st.session_state.query_history.append({
                            "query": user_query,
                            "timestamp": datetime.now().isoformat()
                        })
                        
                        logger.info("Query processed successfully")
                    
                    # Display the response
                    st.markdown("---")
                    st.subheader("📈 Analysis Results")
                    st.markdown(response_content)
                    
                    # Export options
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("📥 Download Results as Text"):
                            st.download_button(
                                label="Download Text",
                                data=response_content,
                                file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                                mime="text/plain"
                            )
                    
                    with col2:
                        if st.button("💾 Save to History"):
                            st.success("✅ Result saved to query history")
                    
                except Exception as e:
                    logger.error(f"Query execution failed: {e}")
                    st.error(f"❌ Error generating response: {str(e)}")
                    st.error("💡 Please try rephrasing your query or check if the data format is correct.")

elif uploaded_file is not None and not st.session_state.openai_key:
    st.error("❌ Please enter your OpenAI API key in the sidebar to proceed.")

else:
    # Welcome message when no file is uploaded
    st.info("""
    ### Welcome to Data Analyst Agent!
    
    **Getting started:**
    1. Enter your OpenAI API key in the sidebar
    2. Upload a CSV or Excel file
    3. Ask questions about your data in natural language
    
    **Features:**
    - 📊 Automatic data profiling
    - 🤖 AI-powered analysis with GPT-4o
    - 🔍 SQL query generation
    - 📥 Export results
    - 📜 Query history tracking
    
    **Example questions:**
    - "What are the summary statistics?"
    - "Show me the top 10 values in column X"
    - "What correlations exist in the data?"
    - "Identify any outliers"
    """)
