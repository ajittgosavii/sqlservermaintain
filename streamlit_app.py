import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
import time
import json
import os
import logging
from typing import Dict, List, Optional, Any

# Configure logging for Streamlit Cloud
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# SQL Server drivers - optional imports with fallbacks
try:
    import pymssql
    PYMSSQL_AVAILABLE = True
except ImportError:
    PYMSSQL_AVAILABLE = False
    pymssql = None

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    sqlalchemy = None

# AI integration - optional
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

# Streamlit Cloud configuration
st.set_page_config(
    page_title="SQL Server AI Optimizer Suite",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS for professional look
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #28a745;
        text-align: center;
    }
    .critical-card {
        border-left-color: #dc3545;
        background: #fff5f5;
    }
    .warning-card {
        border-left-color: #ffc107;
        background: #fffbf0;
    }
    .success-card {
        border-left-color: #28a745;
        background: #f0fff4;
    }
    .ai-insight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .status-green { background-color: #28a745; }
    .status-yellow { background-color: #ffc107; }
    .status-red { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

class StreamlitCloudSQLServerManager:
    """Simplified SQL Server manager optimized for Streamlit Cloud"""
    
    def __init__(self):
        self.connections = {}
        self.load_cloud_configuration()
    
    def load_cloud_configuration(self):
        """Load configuration from Streamlit Cloud secrets"""
        try:
            # Try to load from Streamlit secrets
            if hasattr(st, 'secrets') and 'sql_servers' in st.secrets:
                sql_secrets = st.secrets['sql_servers']
                
                # Support multiple server configurations
                server_configs = {
                    'primary': {
                        'host': sql_secrets.get('primary_host', ''),
                        'port': int(sql_secrets.get('primary_port', 1433)),
                        'username': sql_secrets.get('primary_username', ''),
                        'password': sql_secrets.get('primary_password', ''),
                        'database': sql_secrets.get('primary_database', 'master'),
                        'name': 'Production Server'
                    },
                    'secondary': {
                        'host': sql_secrets.get('secondary_host', ''),
                        'port': int(sql_secrets.get('secondary_port', 1433)),
                        'username': sql_secrets.get('secondary_username', ''),
                        'password': sql_secrets.get('secondary_password', ''),
                        'database': sql_secrets.get('secondary_database', 'master'),
                        'name': 'Analytics Server'
                    },
                    'tertiary': {
                        'host': sql_secrets.get('tertiary_host', ''),
                        'port': int(sql_secrets.get('tertiary_port', 1433)),
                        'username': sql_secrets.get('tertiary_username', ''),
                        'password': sql_secrets.get('tertiary_password', ''),
                        'database': sql_secrets.get('tertiary_database', 'master'),
                        'name': 'Reporting Server'
                    }
                }
                
                # Only add servers with valid configuration
                for server_key, config in server_configs.items():
                    if config['host'] and config['username'] and config['password']:
                        self.connections[config['name']] = config
                        logger.info(f"Loaded SQL Server config: {config['name']} -> {config['host']}")
            
        except Exception as e:
            logger.warning(f"Failed to load SQL Server configuration from secrets: {e}")
            
        # If no configurations loaded, we'll use demo mode
        if not self.connections:
            logger.info("No SQL Server configurations found - will use demo data")
    
    def test_connection(self, server_name: str) -> tuple[bool, str]:
        """Test connection to a specific SQL Server"""
        if server_name not in self.connections:
            return False, "Server configuration not found"
        
        config = self.connections[server_name]
        
        try:
            if PYMSSQL_AVAILABLE:
                conn = pymssql.connect(
                    server=config['host'],
                    port=config['port'],
                    user=config['username'],
                    password=config['password'],
                    database=config['database'],
                    timeout=10
                )
                
                cursor = conn.cursor()
                cursor.execute("SELECT @@VERSION")
                version = cursor.fetchone()[0]
                cursor.close()
                conn.close()
                
                return True, f"Connected successfully! SQL Server version: {version[:50]}..."
            
            elif SQLALCHEMY_AVAILABLE:
                connection_url = (
                    f"mssql+pymssql://{config['username']}:{config['password']}"
                    f"@{config['host']}:{config['port']}/{config['database']}"
                )
                
                engine = create_engine(connection_url, pool_timeout=10)
                with engine.connect() as conn:
                    result = conn.execute(text("SELECT @@VERSION"))
                    version = result.fetchone()[0]
                
                return True, f"Connected successfully! SQL Server version: {version[:50]}..."
            
            else:
                return False, "SQL Server drivers not available. Install pymssql or sqlalchemy."
                
        except Exception as e:
            return False, f"Connection failed: {str(e)}"
    
    def get_performance_data(self, server_name: str) -> pd.DataFrame:
        """Get performance data from SQL Server or generate demo data"""
        if server_name not in self.connections:
            return self._generate_demo_data()
        
        config = self.connections[server_name]
        
        try:
            # SQL Server performance query
            query = """
            SELECT TOP 100
                qs.creation_time as timestamp,
                'sql_server' as application,
                'query_' + CAST(ROW_NUMBER() OVER (ORDER BY qs.total_elapsed_time DESC) AS VARCHAR(10)) as query_id,
                CAST(qs.total_elapsed_time / 1000.0 AS FLOAT) as execution_time_ms,
                CAST(CASE 
                    WHEN qs.total_elapsed_time > 0 
                    THEN (qs.total_worker_time * 100.0) / qs.total_elapsed_time
                    ELSE 0.0 
                END AS FLOAT) as cpu_usage_percent,
                CAST(qs.total_logical_reads AS FLOAT) / NULLIF(qs.execution_count, 0) * 8 / 1024.0 as memory_usage_mb,
                CAST(CASE 
                    WHEN (qs.total_physical_reads + qs.total_logical_reads) > 0
                    THEN (qs.total_logical_reads - qs.total_physical_reads) * 100.0 / 
                        (qs.total_logical_reads + qs.total_physical_reads)
                    ELSE 90.0 
                END AS FLOAT) as cache_hit_ratio,
                qs.execution_count as calls,
                DB_NAME() as database_name,
                qs.total_logical_reads as rows_examined,
                CAST(CASE 
                    WHEN qs.execution_count > 0 
                    THEN qs.total_rows / qs.execution_count 
                    ELSE 1 
                END AS FLOAT) as rows_returned,
                qs.execution_count as connection_id,
                'system' as user_name,
                CASE 
                    WHEN qs.total_elapsed_time > 5000000 THEN 'LONG_QUERY'
                    WHEN qs.total_physical_reads > qs.total_logical_reads * 0.1 THEN 'PAGEIOLATCH_SH'
                    ELSE ''
                END as wait_event
            FROM sys.dm_exec_query_stats qs
            WHERE qs.creation_time > DATEADD(HOUR, -24, GETDATE())
                AND qs.total_elapsed_time > 0
            ORDER BY qs.total_elapsed_time DESC
            """
            
            if PYMSSQL_AVAILABLE:
                conn = pymssql.connect(
                    server=config['host'],
                    port=config['port'],
                    user=config['username'],
                    password=config['password'],
                    database=config['database'],
                    timeout=30
                )
                
                df = pd.read_sql_query(query, conn)
                conn.close()
                
                if not df.empty:
                    # Ensure proper data types
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    numeric_cols = ['execution_time_ms', 'cpu_usage_percent', 'memory_usage_mb', 'cache_hit_ratio']
                    for col in numeric_cols:
                        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    
                    logger.info(f"Retrieved {len(df)} real performance records from {server_name}")
                    return df
                else:
                    logger.warning(f"No recent query activity on {server_name} - using demo data")
                    return self._generate_demo_data()
            
            elif SQLALCHEMY_AVAILABLE:
                connection_url = (
                    f"mssql+pymssql://{config['username']}:{config['password']}"
                    f"@{config['host']}:{config['port']}/{config['database']}"
                )
                
                engine = create_engine(connection_url)
                df = pd.read_sql_query(text(query), engine)
                
                if not df.empty:
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    logger.info(f"Retrieved {len(df)} real performance records from {server_name}")
                    return df
                else:
                    return self._generate_demo_data()
            
            else:
                return self._generate_demo_data()
                
        except Exception as e:
            logger.error(f"Failed to get performance data from {server_name}: {e}")
            return self._generate_demo_data()
    
    def get_database_info(self, server_name: str) -> Dict[str, Any]:
        """Get database information from SQL Server"""
        if server_name not in self.connections:
            return self._generate_demo_stats()
        
        config = self.connections[server_name]
        stats = {}
        
        try:
            if PYMSSQL_AVAILABLE:
                conn = pymssql.connect(
                    server=config['host'],
                    port=config['port'],
                    user=config['username'],
                    password=config['password'],
                    database=config['database'],
                    timeout=10
                )
                
                cursor = conn.cursor()
                
                # Get active connections
                cursor.execute("""
                    SELECT COUNT(*) as active_connections 
                    FROM sys.dm_exec_sessions 
                    WHERE is_user_process = 1 AND status IN ('running', 'sleeping')
                """)
                connections = cursor.fetchone()[0]
                stats['connections'] = connections
                
                # Get database size
                cursor.execute("""
                    SELECT 
                        CAST(SUM(CAST(size as BIGINT)) * 8.0 / 1024 / 1024 AS DECIMAL(10,2)) as size_gb
                    FROM sys.master_files 
                    WHERE database_id = DB_ID()
                """)
                size_gb = cursor.fetchone()[0]
                stats['database_size'] = f"{size_gb:.1f} GB"
                
                # Get cache hit ratio
                cursor.execute("""
                    SELECT TOP 1
                        CAST(cntr_value AS FLOAT) as cache_hit_ratio
                    FROM sys.dm_os_performance_counters 
                    WHERE counter_name LIKE '%Buffer cache hit ratio%' 
                    AND instance_name = ''
                """)
                cache_result = cursor.fetchone()
                if cache_result:
                    stats['cache_hit_ratio'] = cache_result[0]
                else:
                    stats['cache_hit_ratio'] = 90.0
                
                cursor.close()
                conn.close()
                
                logger.info(f"Retrieved database stats from {server_name}: {stats}")
                return stats
                
        except Exception as e:
            logger.error(f"Failed to get database info from {server_name}: {e}")
        
        return self._generate_demo_stats()
    
    def _generate_demo_data(self) -> pd.DataFrame:
        """Generate realistic demo data for SQL Server"""
        logger.info("Generating demo SQL Server performance data")
        
        base_time = datetime.now() - timedelta(hours=24)
        
        # Realistic SQL Server application patterns
        applications = [
            {"name": "web_api", "base_time": 150, "variance": 50, "volume": 0.4},
            {"name": "mobile_api", "base_time": 200, "variance": 80, "volume": 0.3},
            {"name": "batch_processor", "base_time": 2000, "variance": 500, "volume": 0.2},
            {"name": "analytics_engine", "base_time": 5000, "variance": 2000, "volume": 0.1}
        ]
        
        data = []
        for i in range(1500):  # Generate realistic amount of data
            app_index = np.random.choice(len(applications), p=[a["volume"] for a in applications])
            app_config = applications[app_index]
            app_name = app_config["name"]
            
            timestamp = base_time + timedelta(seconds=np.random.randint(0, 86400))
            
            # Business hours effect
            hour = timestamp.hour
            business_hours_multiplier = 1.5 if 9 <= hour <= 17 else 0.7
            
            exec_time = max(10, np.random.normal(
                app_config["base_time"] * business_hours_multiplier, 
                app_config["variance"]
            ))
            
            data.append({
                "timestamp": timestamp,
                "application": app_name,
                "query_id": f"q_{i % 150}",
                "execution_time_ms": exec_time,
                "cpu_usage_percent": min(100, max(0, exec_time / 40 + np.random.normal(0, 15))),
                "memory_usage_mb": max(10, np.random.normal(300, 150)),
                "rows_examined": max(1, int(np.random.exponential(2000))),
                "rows_returned": max(1, int(np.random.exponential(200))),
                "cache_hit_ratio": np.random.uniform(0.65, 0.98),
                "connection_id": np.random.randint(1, 100),
                "database_name": "demo_database",
                "user_name": f"{app_name}_user",
                "wait_event": np.random.choice(["", "PAGEIOLATCH_SH", "LCK_M_S", "WRITELOG"], p=[0.7, 0.1, 0.15, 0.05]),
                "calls": max(1, int(np.random.exponential(10)))
            })
        
        return pd.DataFrame(data)
    
    def _generate_demo_stats(self) -> Dict[str, Any]:
        """Generate demo database statistics"""
        return {
            "connections": np.random.randint(20, 80),
            "database_size": f"{np.random.uniform(100, 500):.1f} GB",
            "cache_hit_ratio": np.random.uniform(85, 95),
            "longest_query": np.random.uniform(0, 30)
        }

class StreamlitCloudAIManager:
    """Simplified AI manager for Streamlit Cloud with Anthropic Claude"""
    
    def __init__(self):
        self.client = None
        self.available = False
        self._initialize_claude()
    
    def _initialize_claude(self):
        """Initialize Claude AI client from Streamlit secrets"""
        if not ANTHROPIC_AVAILABLE:
            logger.warning("Anthropic library not available")
            return
        
        try:
            # Try to get API key from Streamlit secrets
            api_key = None
            if hasattr(st, 'secrets') and 'ANTHROPIC_API_KEY' in st.secrets:
                api_key = st.secrets['ANTHROPIC_API_KEY']
            elif 'ANTHROPIC_API_KEY' in os.environ:
                api_key = os.environ['ANTHROPIC_API_KEY']
            
            if api_key:
                self.client = anthropic.Anthropic(api_key=api_key)
                self.available = True
                logger.info("Claude AI initialized successfully")
            else:
                logger.warning("Claude AI API key not found in secrets or environment")
                
        except Exception as e:
            logger.error(f"Failed to initialize Claude AI: {e}")
    
    def analyze_sql_performance(self, data: pd.DataFrame, query_type: str = "overview") -> str:
        """Analyze SQL Server performance data with Claude AI"""
        if not self.available or data.empty:
            return self._generate_fallback_analysis(data, query_type)
        
        try:
            # Prepare data summary for AI
            summary = self._prepare_data_summary(data)
            
            prompt = f"""
You are a senior SQL Server DBA analyzing enterprise database performance. Based on this performance data, provide actionable insights:

{summary}

Please provide a {query_type} analysis with:

1. **Key Performance Insights** (2-3 bullet points)
2. **SQL Server Optimization Recommendations** (2-3 specific actions)
3. **Risk Assessment** (1-2 sentences)

Focus on SQL Server-specific optimizations like index tuning, query plan optimization, and resource configuration.
Keep response under 300 words and make recommendations actionable for a SQL Server DBA.
"""
            
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            logger.error(f"Claude AI analysis failed: {e}")
            return self._generate_fallback_analysis(data, query_type)
    
    def _prepare_data_summary(self, data: pd.DataFrame) -> str:
        """Prepare concise data summary for AI analysis"""
        if data.empty:
            return "No performance data available"
        
        # Calculate key metrics
        avg_time = data['execution_time_ms'].mean()
        p95_time = data['execution_time_ms'].quantile(0.95)
        slow_queries = (data['execution_time_ms'] > 5000).sum()
        total_queries = len(data)
        
        avg_cpu = data['cpu_usage_percent'].mean()
        avg_memory = data['memory_usage_mb'].mean()
        avg_cache = data['cache_hit_ratio'].mean()
        
        applications = data['application'].value_counts()
        
        summary = f"""
SQL Server Performance Summary:
- Total Queries: {total_queries:,}
- Average Response Time: {avg_time:.1f}ms
- 95th Percentile: {p95_time:.1f}ms
- Slow Queries (>5s): {slow_queries} ({slow_queries/total_queries*100:.1f}%)
- Average CPU: {avg_cpu:.1f}%
- Average Memory: {avg_memory:.1f}MB
- Buffer Cache Hit Ratio: {avg_cache:.1f}%
- Top Applications: {', '.join(applications.head(3).index.tolist())}
"""
        return summary
    
    def _generate_fallback_analysis(self, data: pd.DataFrame, query_type: str) -> str:
        """Generate statistical analysis when AI is not available"""
        if data.empty:
            return "No performance data available for analysis."
        
        avg_time = data['execution_time_ms'].mean()
        slow_queries = (data['execution_time_ms'] > 5000).sum()
        total_queries = len(data)
        slow_rate = (slow_queries / total_queries * 100) if total_queries > 0 else 0
        
        if slow_rate > 10:
            status = "🔴 **Critical Performance Issues**"
            recommendation = "Immediate action required: Review slow queries, check for missing indexes, analyze wait statistics"
        elif slow_rate > 5:
            status = "🟡 **Performance Degradation**"
            recommendation = "Performance tuning needed: Use Query Store, review execution plans, update statistics"
        else:
            status = "🟢 **Good Performance**"
            recommendation = "Maintain current performance: Regular monitoring, proactive maintenance"
        
        return f"""
**Statistical Analysis Results:**

{status}

**Key Metrics:**
• Average response time: {avg_time:.1f}ms
• Slow query rate: {slow_rate:.1f}% ({slow_queries}/{total_queries})
• Buffer cache performance: {data['cache_hit_ratio'].mean():.1f}%

**Recommendations:**
{recommendation}

*Note: Connect Claude AI for detailed analysis and specific optimization recommendations.*
"""

# Initialize managers
@st.cache_resource
def get_sql_manager():
    return StreamlitCloudSQLServerManager()

@st.cache_resource  
def get_ai_manager():
    return StreamlitCloudAIManager()

def main():
    """Main application for Streamlit Cloud"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🗄️ SQL Server AI Optimizer Suite</h1>
        <p><strong>Enterprise Database Performance Analytics for Streamlit Cloud</strong></p>
        <p>🤖 AI-Powered • ☁️ Cloud-Ready • 🔒 Secure</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize managers
    sql_manager = get_sql_manager()
    ai_manager = get_ai_manager()
    
    # Show connection status
    show_connection_status(sql_manager, ai_manager)
    
    # Main navigation
    page = st.sidebar.selectbox(
        "🗄️ SQL Server Analytics",
        [
            "🏠 Executive Dashboard",
            "⚡ Performance Analysis", 
            "🗄️ Database Health",
            "🤖 AI Insights",
            "⚙️ Configuration",
            "📚 Setup Guide"
        ]
    )
    
    # Route to selected page
    if page == "🏠 Executive Dashboard":
        show_executive_dashboard(sql_manager, ai_manager)
    elif page == "⚡ Performance Analysis":
        show_performance_analysis(sql_manager, ai_manager)
    elif page == "🗄️ Database Health":
        show_database_health(sql_manager)
    elif page == "🤖 AI Insights":
        show_ai_insights(sql_manager, ai_manager)
    elif page == "⚙️ Configuration":
        show_configuration(sql_manager)
    elif page == "📚 Setup Guide":
        show_setup_guide()

def show_connection_status(sql_manager, ai_manager):
    """Show current connection status"""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # SQL Server status
        if sql_manager.connections:
            connected_servers = []
            for server_name in sql_manager.connections.keys():
                success, message = sql_manager.test_connection(server_name)
                if success:
                    connected_servers.append(server_name)
            
            if connected_servers:
                st.success(f"🗄️ Connected to {len(connected_servers)} SQL Server(s): {', '.join(connected_servers)}")
            else:
                st.warning("🗄️ SQL Server connections configured but not accessible - using demo data")
        else:
            st.info("🗄️ No SQL Server connections configured - using demo data")
    
    with col2:
        # AI status
        if ai_manager.available:
            st.success("🤖 Claude AI: Ready")
        else:
            st.warning("🤖 AI: Configure API key")

def show_executive_dashboard(sql_manager, ai_manager):
    """Executive dashboard with key metrics"""
    st.header("🏠 Executive Performance Dashboard")
    
    # Data source selection
    data_source = "Demo Data"
    if sql_manager.connections:
        data_source = st.selectbox(
            "Data Source",
            ["Demo Data"] + list(sql_manager.connections.keys())
        )
    
    # Load performance data
    with st.spinner(f"Loading performance data from {data_source}..."):
        if data_source == "Demo Data":
            performance_data = sql_manager._generate_demo_data()
        else:
            performance_data = sql_manager.get_performance_data(data_source)
    
    if performance_data.empty:
        st.error("No performance data available")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_time = performance_data['execution_time_ms'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Response Time", f"{avg_time:.0f}ms", f"{np.random.uniform(-50, 50):.0f}ms")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        total_queries = len(performance_data)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Queries", f"{total_queries:,}", f"+{np.random.randint(100, 500)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        slow_queries = (performance_data['execution_time_ms'] > 5000).sum()
        slow_rate = (slow_queries / total_queries * 100) if total_queries > 0 else 0
        card_class = "critical-card" if slow_rate > 10 else "warning-card" if slow_rate > 5 else "success-card"
        st.markdown(f'<div class="metric-card {card_class}">', unsafe_allow_html=True)
        st.metric("Slow Query Rate", f"{slow_rate:.1f}%", f"{np.random.uniform(-1, 1):.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        avg_cache = performance_data['cache_hit_ratio'].mean()
        st.markdown('<div class="metric-card success-card">', unsafe_allow_html=True)
        st.metric("Cache Hit Ratio", f"{avg_cache:.1f}%", f"+{np.random.uniform(0, 2):.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Performance charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Response time trend
        hourly_data = performance_data.groupby(performance_data['timestamp'].dt.hour)['execution_time_ms'].mean()
        fig1 = px.line(
            x=hourly_data.index, 
            y=hourly_data.values,
            title="Response Time by Hour",
            labels={'x': 'Hour', 'y': 'Response Time (ms)'}
        )
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Application performance
        app_perf = performance_data.groupby('application')['execution_time_ms'].mean().sort_values(ascending=True)
        fig2 = px.bar(
            x=app_perf.values, 
            y=app_perf.index, 
            orientation='h',
            title="Performance by Application",
            labels={'x': 'Avg Response Time (ms)', 'y': 'Application'}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Quick AI insight
    if ai_manager.available:
        if st.button("🚀 Generate AI Executive Summary"):
            with st.spinner("🤖 Analyzing performance data with Claude AI..."):
                ai_analysis = ai_manager.analyze_sql_performance(performance_data, "executive")
                st.markdown(f'<div class="ai-insight">{ai_analysis}</div>', unsafe_allow_html=True)

def show_performance_analysis(sql_manager, ai_manager):
    """Detailed performance analysis"""
    st.header("⚡ SQL Server Performance Analysis")
    
    # Data source selection
    data_source = "Demo Data"
    if sql_manager.connections:
        data_source = st.selectbox(
            "Select SQL Server",
            ["Demo Data"] + list(sql_manager.connections.keys()),
            key="perf_analysis_source"
        )
    
    # Load data
    if data_source == "Demo Data":
        data = sql_manager._generate_demo_data()
    else:
        data = sql_manager.get_performance_data(data_source)
    
    if data.empty:
        st.error("No performance data available")
        return
    
    # Performance distribution
    st.subheader("📊 Query Performance Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Execution time histogram
        fig1 = px.histogram(
            data, 
            x='execution_time_ms', 
            nbins=50,
            title="Execution Time Distribution"
        )
        fig1.add_vline(x=5000, line_dash="dash", line_color="red", annotation_text="5s Threshold")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # CPU vs Response Time scatter
        fig2 = px.scatter(
            data, 
            x='cpu_usage_percent', 
            y='execution_time_ms',
            color='application',
            title="CPU Usage vs Response Time"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Slow query analysis
    st.subheader("🔍 Slow Query Analysis")
    
    slow_threshold = st.slider("Slow Query Threshold (ms)", 1000, 10000, 5000)
    slow_queries = data[data['execution_time_ms'] > slow_threshold]
    
    if not slow_queries.empty:
        st.warning(f"Found {len(slow_queries)} queries exceeding {slow_threshold}ms")
        
        # Slow queries by application
        slow_by_app = slow_queries.groupby('application').agg({
            'execution_time_ms': ['count', 'mean', 'max'],
            'cpu_usage_percent': 'mean'
        }).round(2)
        
        slow_by_app.columns = ['Count', 'Avg Time (ms)', 'Max Time (ms)', 'Avg CPU %']
        st.dataframe(slow_by_app, use_container_width=True)
        
        # AI analysis of slow queries
        if ai_manager.available and st.button("🤖 Analyze Slow Queries with AI"):
            with st.spinner("🤖 Analyzing slow query patterns..."):
                ai_analysis = ai_manager.analyze_sql_performance(slow_queries, "slow_query")
                st.markdown(f'<div class="ai-insight">{ai_analysis}</div>', unsafe_allow_html=True)
    else:
        st.success(f"✅ No queries exceed {slow_threshold}ms threshold")

def show_database_health(sql_manager):
    """Database health monitoring"""
    st.header("🗄️ SQL Server Database Health")
    
    # Server selection
    if sql_manager.connections:
        selected_server = st.selectbox(
            "Select SQL Server",
            list(sql_manager.connections.keys()),
            key="health_server"
        )
        
        # Get database info
        db_info = sql_manager.get_database_info(selected_server)
        
        # Health metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Connections", db_info.get('connections', 'N/A'))
        
        with col2:
            st.metric("Database Size", db_info.get('database_size', 'N/A'))
        
        with col3:
            cache_ratio = db_info.get('cache_hit_ratio', 0)
            st.metric("Buffer Cache Hit", f"{cache_ratio:.1f}%")
        
        with col4:
            st.metric("Status", "🟢 Healthy" if cache_ratio > 80 else "🟡 Monitor")
        
        # Connection test
        if st.button(f"🔍 Test Connection to {selected_server}"):
            success, message = sql_manager.test_connection(selected_server)
            if success:
                st.success(f"✅ {message}")
            else:
                st.error(f"❌ {message}")
    
    else:
        st.info("Configure SQL Server connections to view real health metrics")
        
        # Demo health metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Demo Connections", random.randint(20, 80))
        
        with col2:
            st.metric("Demo Database Size", f"{random.uniform(100, 500):.1f} GB")
        
        with col3:
            st.metric("Demo Cache Hit", f"{random.uniform(85, 95):.1f}%")
        
        with col4:
            st.metric("Demo Status", "🟢 Healthy")

def show_ai_insights(sql_manager, ai_manager):
    """AI-powered insights and recommendations"""
    st.header("🤖 AI-Powered SQL Server Insights")
    
    if not ai_manager.available:
        st.error("🔑 Claude AI not configured. Add your ANTHROPIC_API_KEY to Streamlit secrets.")
        st.info("Go to your Streamlit Cloud app settings and add the secret: ANTHROPIC_API_KEY")
        return
    
    # Data source
    data_source = "Demo Data"
    if sql_manager.connections:
        data_source = st.selectbox(
            "Data Source for AI Analysis",
            ["Demo Data"] + list(sql_manager.connections.keys()),
            key="ai_source"
        )
    
    # Load data
    if data_source == "Demo Data":
        data = sql_manager._generate_demo_data()
    else:
        data = sql_manager.get_performance_data(data_source)
    
    if data.empty:
        st.error("No data available for AI analysis")
        return
    
    # Analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        analysis_type = st.selectbox(
            "Analysis Type",
            [
                "overview",
                "performance_tuning", 
                "slow_query",
                "capacity_planning",
                "security_review"
            ]
        )
    
    with col2:
        if st.button("🚀 Generate AI Analysis", type="primary"):
            with st.spinner("🤖 Claude AI is analyzing your SQL Server performance..."):
                ai_analysis = ai_manager.analyze_sql_performance(data, analysis_type)
                st.markdown(f'<div class="ai-insight">{ai_analysis}</div>', unsafe_allow_html=True)
    
    # Show data summary for context
    st.subheader("📊 Data Summary for AI Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Queries", len(data))
        st.metric("Avg Response Time", f"{data['execution_time_ms'].mean():.0f}ms")
    
    with col2:
        slow_queries = (data['execution_time_ms'] > 5000).sum()
        st.metric("Slow Queries", slow_queries)
        st.metric("Applications", data['application'].nunique())
    
    with col3:
        st.metric("Avg CPU Usage", f"{data['cpu_usage_percent'].mean():.1f}%")
        st.metric("Cache Hit Ratio", f"{data['cache_hit_ratio'].mean():.1f}%")

def show_configuration(sql_manager):
    """Configuration page for Streamlit Cloud"""
    st.header("⚙️ Configuration")
    
    # Connection status
    st.subheader("🗄️ SQL Server Connections")
    
    if sql_manager.connections:
        st.success(f"✅ {len(sql_manager.connections)} SQL Server(s) configured")
        
        for server_name, config in sql_manager.connections.items():
            with st.expander(f"📊 {server_name}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.text(f"Host: {config['host']}")
                    st.text(f"Port: {config['port']}")
                    st.text(f"Database: {config['database']}")
                
                with col2:
                    st.text(f"Username: {config['username']}")
                    if st.button(f"🔍 Test {server_name}"):
                        success, message = sql_manager.test_connection(server_name)
                        if success:
                            st.success(f"✅ {message}")
                        else:
                            st.error(f"❌ {message}")
    else:
        st.warning("⚠️ No SQL Server connections configured")
    
    # Configuration guide
    st.subheader("📚 Configuration Guide")
    
    st.markdown("""
    ### 🔧 Streamlit Cloud Setup
    
    **1. SQL Server Configuration**
    
    Add these secrets to your Streamlit Cloud app:
    
    ```toml
    [sql_servers]
    # Primary SQL Server
    primary_host = "your-ec2-ip-1"
    primary_port = 1433
    primary_username = "db_monitor"
    primary_password = "your-secure-password"
    primary_database = "YourDatabase"
    
    # Secondary SQL Server (optional)
    secondary_host = "your-ec2-ip-2"
    secondary_port = 1433
    secondary_username = "db_monitor"
    secondary_password = "your-secure-password"
    secondary_database = "YourDatabase"
    
    # Third SQL Server (optional)
    tertiary_host = "your-ec2-ip-3"
    tertiary_port = 1433
    tertiary_username = "db_monitor"
    tertiary_password = "your-secure-password"
    tertiary_database = "YourDatabase"
    ```
    
    **2. Claude AI Configuration**
    
    ```toml
    ANTHROPIC_API_KEY = "sk-ant-api03-your-api-key"
    ```
    
    **3. Security Notes**
    - Use dedicated monitoring user with read-only permissions
    - Ensure EC2 security groups allow access from Streamlit Cloud
    - Never commit secrets to your repository
    """)

def show_setup_guide():
    """Complete setup guide for Streamlit Cloud"""
    st.header("📚 Complete Setup Guide")
    
    st.markdown("""
    # 🚀 SQL Server AI Optimizer Setup for Streamlit Cloud
    
    ## 📋 Prerequisites
    
    1. **SQL Server instances** running on AWS EC2
    2. **Streamlit Cloud account** (free at share.streamlit.io)
    3. **Claude AI API key** (optional, for AI features)
    
    ## 🔧 Step 1: Prepare SQL Server
    
    ### Create Monitoring User:
    ```sql
    -- Create dedicated monitoring user
    CREATE LOGIN [db_monitor] WITH PASSWORD = 'SecurePassword123!'
    CREATE USER [db_monitor] FOR LOGIN [db_monitor]
    
    -- Grant monitoring permissions
    GRANT VIEW SERVER STATE TO [db_monitor]
    GRANT VIEW DATABASE STATE TO [db_monitor]
    GRANT VIEW ANY DEFINITION TO [db_monitor]
    
    -- For specific databases
    USE [YourDatabase]
    GRANT SELECT TO [db_monitor]
    ```
    
    ### Configure SQL Server:
    ```sql
    -- Enable remote connections
    EXEC sp_configure 'remote access', 1
    RECONFIGURE
    
    -- Enable TCP/IP (use SQL Server Configuration Manager)
    -- Set port to 1433
    ```
    
    ### AWS Security Group:
    ```bash
    Type: Custom TCP
    Port: 1433
    Source: 0.0.0.0/0  # For Streamlit Cloud access
    Description: SQL Server for Streamlit Cloud monitoring
    ```
    
    ## ☁️ Step 2: Deploy to Streamlit Cloud
    
    ### 1. Push Code to GitHub:
    ```bash
    git add .
    git commit -m "SQL Server monitoring app"
    git push origin main
    ```
    
    ### 2. Deploy on Streamlit Cloud:
    - Go to [share.streamlit.io](https://share.streamlit.io)
    - Connect your GitHub repository
    - Select `streamlit_app.py` as main file
    - Deploy!
    
    ### 3. Configure Secrets:
    In your Streamlit Cloud app settings, add these secrets:
    
    ```toml
    [sql_servers]
    primary_host = "your-ec2-public-ip"
    primary_port = 1433
    primary_username = "db_monitor"
    primary_password = "SecurePassword123!"
    primary_database = "YourDatabase"
    
    # Add secondary and tertiary servers as needed
    
    # Claude AI (optional)
    ANTHROPIC_API_KEY = "sk-ant-api03-your-key"
    ```
    
    ## 🔍 Step 3: Test Connections
    
    1. Open your deployed app
    2. Go to "Configuration" page
    3. Test each SQL Server connection
    4. Verify real data appears in dashboards
    
    ## 📊 Step 4: Monitor Performance
    
    Your app will now show:
    - Real SQL Server performance metrics
    - Live query statistics
    - AI-powered optimization recommendations
    - Database health monitoring
    
    ## 🛠️ Troubleshooting
    
    **Connection Issues:**
    - Check EC2 security group allows port 1433
    - Verify SQL Server is running and accessible
    - Test credentials manually
    
    **Permission Issues:**
    - Grant VIEW SERVER STATE permission
    - Ensure user has database access
    
    **Streamlit Cloud Issues:**
    - Check app logs for errors
    - Verify secrets are configured correctly
    - Restart app if needed
    
    ## 🚀 Success!
    
    Once configured, you'll have:
    ✅ Real-time SQL Server monitoring
    ✅ AI-powered performance insights  
    ✅ Cloud-based analytics dashboard
    ✅ Professional executive reporting
    """)

if __name__ == "__main__":
    main()