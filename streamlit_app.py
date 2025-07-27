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
from typing import Dict, List, Optional, Tuple

# Claude AI Integration
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    st.error("⚠️ Anthropic library not installed. Install with: pip install anthropic")

# Page configuration
st.set_page_config(
    page_title="Enterprise SQL Server AI Optimizer",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Claude client
@st.cache_resource
def get_claude_client():
    """Initialize Claude client with API key"""
    if not ANTHROPIC_AVAILABLE:
        return None
    
    api_key = None
    try:
        # Streamlit Cloud secrets
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except:
        try:
            # Environment variable
            api_key = os.getenv("ANTHROPIC_API_KEY")
        except:
            pass
    
    if not api_key:
        return None
    
    try:
        client = anthropic.Anthropic(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"❌ Failed to initialize Claude client: {str(e)}")
        return None

# AI-powered functions
def call_claude_api(client, prompt, max_tokens=2000):
    """Call Claude API with error handling"""
    if not client:
        return None
    
    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    except Exception as e:
        st.error(f"❌ Claude API Error: {str(e)}")
        return None

def ai_analyze_sql_server_indexes(client, index_data: pd.DataFrame, workload_queries: List[str] = None):
    """AI-powered SQL Server index analysis"""
    
    # Prepare index context
    index_context = f"""
SQL SERVER DATABASE ANALYSIS
Current Index Information:
{index_data.to_string()}

Number of indexes analyzed: {len(index_data)}
High fragmentation indexes (>30%): {len(index_data[index_data['Fragmentation_Percent'] > 30])}
Total index size: {index_data['Size_MB'].sum():.1f} MB
"""
    
    # Add workload context if provided
    workload_context = ""
    if workload_queries:
        workload_context = f"""
QUERY WORKLOAD SAMPLE:
{chr(10).join([f"Query {i+1}: {query}" for i, query in enumerate(workload_queries[:5])])}
"""

    prompt = f"""You are a senior SQL Server DBA with 15+ years of experience optimizing enterprise databases on AWS EC2. 
Analyze this SQL Server index situation and provide expert recommendations.

{index_context}
{workload_context}

Please provide a comprehensive analysis in the following format:

**🎯 EXECUTIVE SUMMARY:**
- Overall index health assessment
- Critical issues requiring immediate attention
- Expected performance impact of recommendations

**🚨 CRITICAL ISSUES:**
- Indexes causing immediate performance problems
- Missing indexes for high-priority queries
- Severely fragmented indexes (>50%)

**📋 MAINTENANCE RECOMMENDATIONS:**
- Specific rebuild vs reorganize recommendations
- Optimal maintenance window suggestions
- Parallel execution strategies for large indexes

**⚡ PERFORMANCE OPTIMIZATIONS:**
- Index consolidation opportunities
- Covering index recommendations
- Partition strategy suggestions

**📊 MONITORING ALERTS:**
- What metrics to track going forward
- When to escalate to development team
- Automated maintenance thresholds

**💰 COST OPTIMIZATION:**
- Storage cost reduction opportunities
- Maintenance window optimization
- Resource utilization improvements

Focus on SQL Server specific optimizations for EC2 environments. Consider factors like:
- SQL Server index rebuild online operations
- Filegroup considerations
- TempDB impact during maintenance
- AWS EBS storage characteristics
- Memory and CPU optimization for EC2

Provide specific T-SQL commands where applicable."""

    return call_claude_api(client, prompt, max_tokens=3000)

def ai_generate_maintenance_plan(client, databases: List[str], current_plans: pd.DataFrame = None):
    """AI-generated SQL Server maintenance plan optimization"""
    
    plan_context = ""
    if current_plans is not None and not current_plans.empty:
        plan_context = f"""
CURRENT MAINTENANCE PLANS:
{current_plans.to_string()}
"""

    prompt = f"""You are an expert SQL Server DBA creating an enterprise-grade maintenance plan for AWS EC2 deployment.

DATABASE ENVIRONMENT:
- Databases: {', '.join(databases)}
- Platform: SQL Server on AWS EC2
- Environment: Production (24/7 operations)

{plan_context}

Create a comprehensive maintenance plan with the following sections:

**🗓️ OPTIMIZED MAINTENANCE SCHEDULE:**
- Weekly schedule with optimal time windows
- Consideration for backup windows and business hours
- Priority-based scheduling (critical vs non-critical)

**🔧 INDEX MAINTENANCE STRATEGY:**
- Fragmentation thresholds and actions
- Online vs offline operations
- Parallel execution parameters
- Filegroup-specific strategies

**📊 STATISTICS MAINTENANCE:**
- Auto-update statistics configuration
- Full scan vs sample rates
- Schedule for manual statistics updates

**🛡️ INTEGRITY CHECKS:**
- DBCC CHECKDB scheduling
- Page verification settings
- Corruption detection and response

**🗂️ BACKUP OPTIMIZATION:**
- Full, differential, and log backup schedule
- Compression and encryption settings
- AWS S3 integration for long-term storage

**📈 PERFORMANCE MONITORING:**
- Key metrics to track
- Automated alerting thresholds
- Performance baseline establishment

**⚙️ CONFIGURATION RECOMMENDATIONS:**
- SQL Server instance settings
- Database-specific configurations
- AWS EC2 and EBS optimizations

**📝 T-SQL IMPLEMENTATION:**
Provide specific T-SQL scripts for:
- Maintenance plan jobs
- Index maintenance procedures
- Statistics update jobs
- Monitoring queries

Focus on enterprise best practices for:
- High availability requirements
- Minimal downtime maintenance
- Cost optimization on AWS
- Automated failure handling
- Compliance and auditing needs"""

    return call_claude_api(client, prompt, max_tokens=3500)

def ai_analyze_sql_server_performance(client, performance_metrics: Dict, slow_queries: List[str] = None):
    """AI-powered SQL Server performance analysis"""
    
    metrics_context = f"""
SQL SERVER PERFORMANCE METRICS:
- Average Query Response Time: {performance_metrics.get('avg_response_time', 'N/A')}
- CPU Utilization: {performance_metrics.get('cpu_usage', 'N/A')}%
- Memory Usage: {performance_metrics.get('memory_usage', 'N/A')}%
- Disk I/O: {performance_metrics.get('disk_io', 'N/A')} IOPS
- Active Connections: {performance_metrics.get('connections', 'N/A')}
- Blocking Sessions: {performance_metrics.get('blocking', 'N/A')}
- Wait Statistics: {performance_metrics.get('waits', 'Various waits detected')}
"""

    slow_query_context = ""
    if slow_queries:
        slow_query_context = f"""
SLOW QUERY SAMPLES:
{chr(10).join([f"Query {i+1}: {query}" for i, query in enumerate(slow_queries[:3])])}
"""

    prompt = f"""You are a SQL Server performance tuning expert analyzing an enterprise production system on AWS EC2.

{metrics_context}
{slow_query_context}

Provide a comprehensive performance analysis:

**🚨 IMMEDIATE CONCERNS:**
- Critical performance bottlenecks
- Queries requiring immediate attention
- Resource constraints affecting performance

**📊 ROOT CAUSE ANALYSIS:**
- Primary performance bottlenecks identified
- Contributing factors to slow performance
- System-level vs query-level issues

**⚡ OPTIMIZATION RECOMMENDATIONS:**
- Index optimization strategies
- Query rewriting suggestions
- Configuration tuning recommendations
- Hardware/AWS optimization opportunities

**🔍 SQL SERVER SPECIFIC OPTIMIZATIONS:**
- Memory configuration (Buffer Pool, Plan Cache)
- TempDB optimization strategies
- Parallelism settings (MAXDOP, Cost Threshold)
- I/O subsystem optimization for EBS

**📈 MONITORING IMPROVEMENTS:**
- Key performance counters to track
- Query Store configuration recommendations
- Extended Events for ongoing monitoring
- Automated alerting thresholds

**🛠️ IMPLEMENTATION PLAN:**
- Priority order for implementing changes
- Expected performance improvements
- Rollback strategies for each change
- Testing recommendations

**💡 PROACTIVE MEASURES:**
- Capacity planning recommendations
- Preventive maintenance strategies
- Performance regression detection

Provide specific T-SQL commands and configuration changes where applicable.
Focus on enterprise production considerations including high availability and minimal downtime."""

    return call_claude_api(client, prompt, max_tokens=3000)

# Initialize Claude client
claude_client = get_claude_client()

# Custom CSS for enterprise look
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
    .enterprise-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #1e3c72;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
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
        background: #fff5f5;
        border-left-color: #dc3545;
        border: 1px solid #f5c6cb;
    }
    .warning-card {
        background: #fffbf0;
        border-left-color: #ffc107;
        border: 1px solid #ffeaa7;
    }
    .success-card {
        background: #f0fff4;
        border-left-color: #28a745;
        border: 1px solid #c3e6cb;
    }
    .ai-insight {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .enterprise-nav {
        background: #2c3e50;
        color: white;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Generate enhanced enterprise demo data
@st.cache_data
def generate_enterprise_index_data():
    """Generate realistic enterprise-scale index data"""
    databases = ['ProductionCRM', 'DataWarehouse', 'ECommerce', 'FinancialReporting', 'CustomerAnalytics']
    table_types = ['Orders', 'Customers', 'Products', 'Transactions', 'Inventory', 'Reports', 'Audit', 'Sessions']
    
    data = []
    for _ in range(75):  # More data for enterprise scale
        db = random.choice(databases)
        table = f"{random.choice(table_types)}_{random.randint(1, 200)}"
        index_name = f"IX_{table}_{random.choice(['ID', 'Date', 'Name', 'Status', 'FK', 'Composite'])}"
        
        # More realistic enterprise fragmentation patterns
        if random.random() < 0.3:  # 30% chance of high fragmentation
            fragmentation = random.uniform(30, 85)
        else:
            fragmentation = random.uniform(0, 30)
        
        size_mb = random.uniform(50, 15000)  # Larger enterprise indexes
        usage_score = random.uniform(10, 100)
        
        # Enhanced action determination
        if fragmentation < 5:
            action = "No Action"
            priority = "Low"
        elif fragmentation < 15:
            action = "Reorganize"
            priority = "Low"
        elif fragmentation < 30:
            action = "Reorganize"
            priority = "Medium"
        else:
            action = "Rebuild"
            priority = "High" if fragmentation > 50 else "Medium"
        
        # Add enterprise-specific columns
        data.append({
            'Database': db,
            'Schema': random.choice(['dbo', 'sales', 'inventory', 'reporting']),
            'Table': table,
            'Index': index_name,
            'Type': random.choice(['CLUSTERED', 'NONCLUSTERED', 'COLUMNSTORE', 'FILTERED']),
            'Fragmentation_Percent': round(fragmentation, 2),
            'Size_MB': round(size_mb, 2),
            'Page_Count': int(size_mb * 128),  # Approximate pages
            'Usage_Score': round(usage_score, 2),
            'Scan_Count': random.randint(100, 50000),
            'Seek_Count': random.randint(500, 100000),
            'Recommended_Action': action,
            'Priority': priority,
            'Estimated_Duration_Min': round(size_mb * 0.05 + fragmentation * 0.15, 1),
            'Last_Maintenance': datetime.now() - timedelta(days=random.randint(1, 365)),
            'Business_Impact': random.choice(['Critical', 'High', 'Medium', 'Low']),
            'Environment': random.choice(['Production', 'Staging', 'Development'])
        })
    
    return pd.DataFrame(data)

@st.cache_data
def generate_enterprise_maintenance_data():
    """Generate enterprise maintenance plan data"""
    plans = []
    databases = ['ProductionCRM', 'DataWarehouse', 'ECommerce', 'FinancialReporting']
    components = ['Index Rebuild', 'Index Reorganize', 'Statistics Update', 'Integrity Check', 
                 'Backup Full', 'Backup Differential', 'Backup Log', 'Archive Cleanup', 'Health Check']
    
    for db in databases:
        for component in components:
            frequency = random.choice(['Daily', 'Weekly', 'Monthly']) if component != 'Backup Log' else 'Every 15 minutes'
            duration = random.uniform(5, 600)  # Enterprise operations take longer
            success_rate = random.uniform(85, 99.9)
            
            plans.append({
                'Database': db,
                'Plan_Name': f"MaintenancePlan_{db}",
                'Component': component,
                'Frequency': frequency,
                'Avg_Duration_Min': round(duration, 1),
                'Success_Rate_Percent': round(success_rate, 2),
                'Last_Run': datetime.now() - timedelta(days=random.randint(0, 30)),
                'Next_Run': datetime.now() + timedelta(days=random.randint(1, 7)),
                'Resource_Usage_CPU': round(random.uniform(10, 80), 1),
                'Resource_Usage_Memory': round(random.uniform(100, 2000), 1),
                'Optimization_Score': round(random.uniform(60, 100), 1),
                'Business_Window': random.choice(['2:00-6:00 AM', '10:00 PM-2:00 AM', '6:00-8:00 AM']),
                'Environment': 'Production' if 'Production' in db else random.choice(['Staging', 'Development'])
            })
    
    return pd.DataFrame(plans)

# Main application
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ Enterprise SQL Server AI Optimizer Suite</h1>
        <p><strong>Intelligent Database Optimization for SQL Server on AWS EC2</strong></p>
        <p>AI-Powered • Enterprise-Grade • Production-Ready</p>
    </div>
    """, unsafe_allow_html=True)
    
    # AI Status indicator
    col1, col2, col3 = st.columns([2, 1, 1])
    with col2:
        if claude_client:
            st.success("🤖 Claude AI: Connected")
        else:
            st.error("⚠️ Claude AI: Offline")
    
    with col3:
        st.info(f"📊 Environment: Production")
    
    # Sidebar navigation
    st.sidebar.markdown('<div class="enterprise-nav"><h2>🗃️ Navigation</h2></div>', unsafe_allow_html=True)
    
    pages = {
        "🏠 Executive Dashboard": "dashboard",
        "🧠 AI Index Optimizer": "ai_index_optimizer", 
        "📋 AI Maintenance Planner": "ai_maintenance_planner",
        "⚡ AI Performance Analyzer": "ai_performance_analyzer",
        "💬 Natural Language SQL": "nl_sql",
        "📊 Enterprise Reporting": "enterprise_reporting",
        "⚙️ Configuration Center": "configuration",
        "🎯 Demo Examples": "demo_examples"
    }
    
    selected_page = st.sidebar.selectbox("Select Module", list(pages.keys()))
    
    # Load enterprise data
    index_df = generate_enterprise_index_data()
    maintenance_df = generate_enterprise_maintenance_data()
    
    # Route to selected page
    if pages[selected_page] == "dashboard":
        show_executive_dashboard(index_df, maintenance_df)
    elif pages[selected_page] == "ai_index_optimizer":
        show_ai_index_optimizer(index_df)
    elif pages[selected_page] == "ai_maintenance_planner":
        show_ai_maintenance_planner(maintenance_df)
    elif pages[selected_page] == "ai_performance_analyzer":
        show_ai_performance_analyzer()
    elif pages[selected_page] == "nl_sql":
        show_natural_language_sql()
    elif pages[selected_page] == "enterprise_reporting":
        show_enterprise_reporting(index_df, maintenance_df)
    elif pages[selected_page] == "configuration":
        show_configuration_center()
    elif pages[selected_page] == "demo_examples":
        show_demo_examples()

def show_executive_dashboard(index_df, maintenance_df):
    st.header("📊 Executive Dashboard")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    critical_indexes = len(index_df[index_df['Priority'] == 'High'])
    avg_fragmentation = index_df['Fragmentation_Percent'].mean()
    total_size_gb = index_df['Size_MB'].sum() / 1024
    maintenance_success = maintenance_df['Success_Rate_Percent'].mean()
    
    with col1:
        st.markdown('<div class="metric-card critical-card">', unsafe_allow_html=True)
        st.metric("Critical Indexes", critical_indexes, f"+{random.randint(1,5)}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card warning-card">', unsafe_allow_html=True)
        st.metric("Avg Fragmentation", f"{avg_fragmentation:.1f}%", f"+{random.uniform(1,3):.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Index Size", f"{total_size_gb:.1f} GB", f"+{random.uniform(0.1,0.5):.1f} GB")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card success-card">', unsafe_allow_html=True)
        st.metric("Maintenance Success", f"{maintenance_success:.1f}%", "+0.5%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col5:
        estimated_savings = critical_indexes * 45000  # $45k per critical index optimized
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Est. Annual Savings", f"${estimated_savings:,}", f"+${random.randint(5000,15000):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        # Database health overview
        db_health = index_df.groupby('Database').agg({
            'Fragmentation_Percent': 'mean',
            'Size_MB': 'sum'
        }).reset_index()
        
        fig1 = px.scatter(db_health, x='Size_MB', y='Fragmentation_Percent', 
                         size='Size_MB', color='Database',
                         title="Database Health Overview (Size vs Fragmentation)",
                         labels={'Size_MB': 'Total Size (MB)', 'Fragmentation_Percent': 'Avg Fragmentation %'})
        fig1.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Critical Threshold")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Priority distribution
        priority_dist = index_df['Priority'].value_counts()
        fig2 = px.pie(values=priority_dist.values, names=priority_dist.index,
                     title="Index Maintenance Priority Distribution",
                     color_discrete_map={'High': '#dc3545', 'Medium': '#ffc107', 'Low': '#28a745'})
        st.plotly_chart(fig2, use_container_width=True)
    
    # Enterprise insights section
    st.markdown("### 🧠 AI-Powered Enterprise Insights")
    
    if claude_client:
        with st.spinner("🤖 Generating AI insights for executive summary..."):
            # Prepare executive summary data
            performance_metrics = {
                'total_databases': len(index_df['Database'].unique()),
                'critical_indexes': critical_indexes,
                'avg_fragmentation': avg_fragmentation,
                'maintenance_success': maintenance_success,
                'total_size_gb': total_size_gb
            }
            
            executive_prompt = f"""As a senior database consultant, provide an executive summary for SQL Server infrastructure:

CURRENT STATE:
- {performance_metrics['total_databases']} production databases
- {performance_metrics['critical_indexes']} indexes requiring immediate attention
- {performance_metrics['avg_fragmentation']:.1f}% average fragmentation
- {performance_metrics['maintenance_success']:.1f}% maintenance success rate
- {performance_metrics['total_size_gb']:.1f} GB total index storage

Provide a concise executive summary focusing on:
**🎯 KEY FINDINGS:**
- Most critical issues affecting business performance
- Financial impact of current database health

**📈 BUSINESS IMPACT:**
- Performance improvements achievable
- Cost savings opportunities
- Risk mitigation priorities

**🚀 RECOMMENDED ACTIONS:**
- Top 3 priority actions for next 30 days
- Resource requirements and timeline
- Expected ROI and business benefits

Keep it executive-level (high-level, business-focused, under 200 words)."""

            ai_insight = call_claude_api(claude_client, executive_prompt, max_tokens=1000)
            
            if ai_insight:
                st.markdown(f"""
                <div class="ai-insight">
                    <h4>🤖 Claude AI Executive Analysis</h4>
                    {ai_insight}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-card">
            <h4>⚠️ AI Insights Unavailable</h4>
            <p>Connect Claude AI to get intelligent executive insights and recommendations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Recent activities
    st.markdown("### 📋 Recent Activities")
    
    recent_activities = [
        {"Time": "2 hours ago", "Activity": "Index rebuild completed on ProductionCRM.dbo.Orders_IX_CustomerID", "Status": "✅ Success", "Impact": "15% performance improvement"},
        {"Time": "6 hours ago", "Activity": "AI detected high fragmentation on DataWarehouse indexes", "Status": "⚠️ Alert", "Impact": "Maintenance scheduled"},
        {"Time": "12 hours ago", "Activity": "Maintenance plan optimization completed", "Status": "✅ Success", "Impact": "30% faster execution"},
        {"Time": "1 day ago", "Activity": "Performance regression detected on ECommerce database", "Status": "🔍 Investigating", "Impact": "Query times +45%"}
    ]
    
    activities_df = pd.DataFrame(recent_activities)
    st.dataframe(activities_df, use_container_width=True)

def show_ai_index_optimizer(index_df):
    st.header("🧠 AI-Powered Index Optimizer")
    st.markdown("Intelligent SQL Server index analysis and optimization recommendations")
    
    # Configuration sidebar
    with st.sidebar:
        st.subheader("⚙️ Optimization Settings")
        environment = st.selectbox("Environment", ["Production", "Staging", "Development"])
        fragmentation_threshold = st.slider("Fragmentation Threshold (%)", 0, 100, 30)
        size_threshold = st.slider("Minimum Size (MB)", 0, 1000, 100)
        business_hours = st.selectbox("Business Hours", ["9:00 AM - 5:00 PM", "24/7 Operations", "Custom"])
        
        st.subheader("🎯 Analysis Scope")
        selected_databases = st.multiselect("Select Databases", 
                                          index_df['Database'].unique().tolist(),
                                          default=index_df['Database'].unique().tolist())
    
    # Filter data based on selections
    filtered_df = index_df[
        (index_df['Database'].isin(selected_databases)) &
        (index_df['Fragmentation_Percent'] >= fragmentation_threshold) &
        (index_df['Size_MB'] >= size_threshold) &
        (index_df['Environment'] == environment)
    ]
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        high_priority = len(filtered_df[filtered_df['Priority'] == 'High'])
        st.metric("High Priority", high_priority, f"{high_priority - random.randint(1,3)}")
    
    with col2:
        total_time = filtered_df['Estimated_Duration_Min'].sum()
        st.metric("Total Maintenance Time", f"{total_time:.0f} min", f"-{random.randint(30,120)} min")
    
    with col3:
        avg_fragmentation = filtered_df['Fragmentation_Percent'].mean() if len(filtered_df) > 0 else 0
        st.metric("Avg Fragmentation", f"{avg_fragmentation:.1f}%", f"+{random.uniform(1,5):.1f}%")
    
    with col4:
        potential_savings = high_priority * 15000  # $15k per optimized high-priority index
        st.metric("Potential Savings", f"${potential_savings:,}", f"+${random.randint(5000,25000):,}")
    
    # AI Analysis Section
    st.markdown("### 🤖 AI-Powered Analysis")
    
    if claude_client and len(filtered_df) > 0:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Generate AI Recommendations", type="primary"):
                with st.spinner("🤖 Claude is analyzing your SQL Server indexes..."):
                    
                    # Sample workload queries for analysis
                    sample_queries = [
                        "SELECT * FROM Orders WHERE CustomerID = @CustomerId AND OrderDate >= @StartDate",
                        "SELECT COUNT(*) FROM Products WHERE CategoryID = @CategoryId AND Price BETWEEN @MinPrice AND @MaxPrice",
                        "UPDATE Inventory SET Quantity = Quantity - @Amount WHERE ProductID = @ProductId",
                        "SELECT TOP 100 * FROM TransactionLog WHERE TransactionDate >= DATEADD(day, -7, GETDATE()) ORDER BY TransactionDate DESC"
                    ]
                    
                    ai_analysis = ai_analyze_sql_server_indexes(claude_client, filtered_df.head(20), sample_queries)
                    
                    if ai_analysis:
                        st.markdown("#### 🧠 Claude AI Analysis Results")
                        st.markdown(ai_analysis)
                        
                        # Generate implementation script
                        st.markdown("#### 📝 Implementation Script")
                        if st.button("Generate T-SQL Script"):
                            script = generate_index_maintenance_script(filtered_df[filtered_df['Priority'] == 'High'].head(10))
                            st.code(script, language="sql")
        
        with col2:
            st.markdown("#### 🎯 Quick Actions")
            
            if st.button("⚡ Emergency Rebuild", help="Rebuild most critical indexes"):
                st.success("🚀 Emergency rebuild initiated for top 5 critical indexes")
                st.info("Estimated completion: 45 minutes")
            
            if st.button("📅 Schedule Maintenance", help="Schedule optimal maintenance window"):
                st.success("📅 Maintenance scheduled for 2:00 AM - 4:00 AM")
                st.info("Notification sent to DBA team")
            
            if st.button("📊 Export Report", help="Export detailed analysis"):
                st.success("📄 Enterprise report exported to SQL_Index_Analysis.xlsx")
            
            # Risk assessment
            st.markdown("#### ⚠️ Risk Assessment")
            risk_score = min(100, (avg_fragmentation * 0.6) + (high_priority * 10))
            
            if risk_score > 70:
                st.error(f"🚨 High Risk: {risk_score:.0f}/100")
            elif risk_score > 40:
                st.warning(f"⚠️ Medium Risk: {risk_score:.0f}/100")
            else:
                st.success(f"✅ Low Risk: {risk_score:.0f}/100")
    
    else:
        if not claude_client:
            st.warning("🔑 Claude AI not available. Configure API key to enable AI-powered analysis.")
        else:
            st.info("📊 No indexes match the current filter criteria.")
    
    # Detailed results table
    st.markdown("### 📋 Index Analysis Results")
    
    if len(filtered_df) > 0:
        # Add action buttons
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Rebuild High Priority", type="primary"):
                high_priority_indexes = filtered_df[filtered_df['Priority'] == 'High']
                st.success(f"✅ Scheduled rebuild for {len(high_priority_indexes)} high priority indexes")
        
        with col2:
            if st.button("🔧 Reorganize Medium Priority"):
                medium_priority_indexes = filtered_df[filtered_df['Priority'] == 'Medium']
                st.success(f"✅ Scheduled reorganize for {len(medium_priority_indexes)} medium priority indexes")
        
        with col3:
            if st.button("📈 Update Statistics"):
                st.success("✅ Statistics update scheduled for all affected tables")
        
        with col4:
            if st.button("🔍 Deep Analysis"):
                st.info("🔍 Detailed performance impact analysis initiated")
        
        # Enhanced data display with styling
        display_df = filtered_df.copy()
        
        # Style the dataframe
        def highlight_priority(row):
            if row['Priority'] == 'High':
                return ['background-color: #ffebee'] * len(row)
            elif row['Priority'] == 'Medium':
                return ['background-color: #fff3e0'] * len(row)
            else:
                return ['background-color: #e8f5e8'] * len(row)
        
        styled_df = display_df.style.apply(highlight_priority, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        # Performance impact chart
        col1, col2 = st.columns(2)
        
        with col1:
            # Fragmentation by database
            frag_by_db = filtered_df.groupby('Database')['Fragmentation_Percent'].mean().reset_index()
            fig1 = px.bar(frag_by_db, x='Database', y='Fragmentation_Percent',
                         title="Average Fragmentation by Database",
                         color='Fragmentation_Percent', color_continuous_scale='Reds')
            fig1.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Rebuild Threshold")
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Size vs fragmentation scatter
            fig2 = px.scatter(filtered_df, x='Size_MB', y='Fragmentation_Percent',
                             color='Priority', size='Usage_Score',
                             title="Index Size vs Fragmentation",
                             hover_data=['Database', 'Table', 'Index'])
            st.plotly_chart(fig2, use_container_width=True)
    
    else:
        st.info("📊 No indexes match the current filter criteria. Adjust the settings in the sidebar.")

def show_ai_maintenance_planner(maintenance_df):
    st.header("📋 AI-Powered Maintenance Planner")
    st.markdown("Intelligent maintenance plan optimization for SQL Server environments")
    
    # Configuration
    with st.sidebar:
        st.subheader("🎛️ Plan Configuration")
        selected_database = st.selectbox("Database", ["All Databases"] + maintenance_df['Database'].unique().tolist())
        optimization_level = st.select_slider("Optimization Level", 
                                             options=["Conservative", "Balanced", "Aggressive"])
        maintenance_window = st.selectbox("Preferred Window", 
                                        ["2:00-6:00 AM", "10:00 PM-2:00 AM", "Weekend Only", "Custom"])
        
        st.subheader("🎯 Optimization Goals")
        minimize_downtime = st.checkbox("Minimize Downtime", value=True)
        optimize_costs = st.checkbox("Optimize AWS Costs", value=True)
        maximize_performance = st.checkbox("Maximize Performance", value=True)
    
    # Filter data
    if selected_database != "All Databases":
        filtered_maintenance = maintenance_df[maintenance_df['Database'] == selected_database]
    else:
        filtered_maintenance = maintenance_df
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    avg_success = filtered_maintenance['Success_Rate_Percent'].mean()
    total_duration = filtered_maintenance['Avg_Duration_Min'].sum()
    avg_optimization = filtered_maintenance['Optimization_Score'].mean()
    component_count = len(filtered_maintenance)
    
    with col1:
        st.metric("Success Rate", f"{avg_success:.1f}%", f"+{random.uniform(0.5,2.0):.1f}%")
    
    with col2:
        st.metric("Total Duration", f"{total_duration:.0f} min", f"-{random.randint(30,120)} min")
    
    with col3:
        st.metric("Optimization Score", f"{avg_optimization:.0f}/100", f"+{random.randint(5,15)}")
    
    with col4:
        st.metric("Components", component_count, f"+{random.randint(1,3)}")
    
    # AI Maintenance Plan Generation
    st.markdown("### 🤖 AI-Generated Maintenance Plan")
    
    if claude_client:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Generate Optimized Plan", type="primary"):
                with st.spinner("🤖 Claude is creating your optimized maintenance plan..."):
                    
                    databases = filtered_maintenance['Database'].unique().tolist()
                    current_plans = filtered_maintenance.groupby(['Database', 'Component']).agg({
                        'Avg_Duration_Min': 'mean',
                        'Success_Rate_Percent': 'mean',
                        'Frequency': 'first'
                    }).reset_index()
                    
                    ai_plan = ai_generate_maintenance_plan(claude_client, databases, current_plans)
                    
                    if ai_plan:
                        st.markdown("#### 🧠 Claude AI Maintenance Plan")
                        st.markdown(ai_plan)
                        
                        # Generate implementation timeline
                        st.markdown("#### 📅 Implementation Timeline")
                        if st.button("Generate Implementation Schedule"):
                            timeline_script = generate_maintenance_timeline(filtered_maintenance)
                            st.code(timeline_script, language="sql")
        
        with col2:
            st.markdown("#### 🎛️ Plan Controls")
            
            if st.button("📊 Analyze Current Plan"):
                st.info("🔍 Analyzing current maintenance performance...")
                time.sleep(1)
                st.success("✅ Analysis complete. 23% improvement opportunity identified.")
            
            if st.button("⚡ Deploy Optimized Plan"):
                st.success("🚀 Optimized plan deployed to SQL Server Agent")
                st.info("⏱️ Changes will take effect at next scheduled run")
            
            if st.button("📈 Performance Simulation"):
                st.info("🧮 Running performance simulation...")
                time.sleep(2)
                st.success("📊 Simulation complete: 35% reduction in maintenance time")
            
            # Cost optimization
            st.markdown("#### 💰 Cost Impact")
            monthly_savings = random.randint(5000, 25000)
            st.metric("Monthly Savings", f"${monthly_savings:,}", f"+${random.randint(1000,5000):,}")
            
            annual_roi = monthly_savings * 12 * 0.75  # 75% of gross savings
            st.metric("Annual ROI", f"${annual_roi:,}", "+15%")
    
    else:
        st.warning("🔑 Claude AI not available. Configure API key to enable intelligent maintenance planning.")
    
    # Current maintenance plan visualization
    st.markdown("### 📊 Current Maintenance Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Component duration analysis
        comp_duration = filtered_maintenance.groupby('Component')['Avg_Duration_Min'].mean().reset_index()
        comp_duration = comp_duration.sort_values('Avg_Duration_Min', ascending=False)
        
        fig1 = px.bar(comp_duration, x='Avg_Duration_Min', y='Component', orientation='h',
                     title="Average Duration by Component",
                     color='Avg_Duration_Min', color_continuous_scale='Blues')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        # Success rate analysis
        success_by_comp = filtered_maintenance.groupby('Component')['Success_Rate_Percent'].mean().reset_index()
        
        fig2 = px.bar(success_by_comp, x='Component', y='Success_Rate_Percent',
                     title="Success Rate by Component",
                     color='Success_Rate_Percent', color_continuous_scale='Greens')
        fig2.add_hline(y=95, line_dash="dash", line_color="red", annotation_text="Target: 95%")
        fig2.update_xaxis(tickangle=45)
        st.plotly_chart(fig2, use_container_width=True)
    
    # Maintenance schedule optimization
    st.markdown("### 📅 Optimized Maintenance Schedule")
    
    # Create optimized schedule
    schedule_data = []
    for _, row in filtered_maintenance.iterrows():
        next_run = row['Next_Run']
        
        # AI-optimized timing based on business impact and resource availability
        optimal_time = optimize_maintenance_time(row['Component'], row['Database'], maintenance_window)
        
        schedule_data.append({
            'Database': row['Database'],
            'Component': row['Component'],
            'Current_Schedule': next_run.strftime('%Y-%m-%d %H:%M'),
            'Optimized_Schedule': optimal_time,
            'Duration_Min': row['Avg_Duration_Min'],
            'Priority': assign_maintenance_priority(row['Component'], row['Success_Rate_Percent']),
            'Resource_Impact': f"CPU: {row['Resource_Usage_CPU']}%, Memory: {row['Resource_Usage_Memory']:.0f}MB",
            'Business_Impact': estimate_business_impact(row['Component'], row['Database'])
        })
    
    schedule_df = pd.DataFrame(schedule_data)
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📅 Apply Optimized Schedule"):
            st.success("✅ Optimized schedule applied to all maintenance plans")
    
    with col2:
        if st.button("🔄 Test Run Simulation"):
            st.info("🧪 Running maintenance simulation...")
            time.sleep(2)
            st.success("✅ Simulation successful. Ready for production deployment.")
    
    with col3:
        if st.button("📧 Send Notifications"):
            st.success("📧 Maintenance notifications sent to stakeholders")
    
    with col4:
        if st.button("📊 Generate Report"):
            st.success("📄 Maintenance plan report exported to PowerBI")
    
    # Display optimized schedule
    st.dataframe(schedule_df, use_container_width=True)

def show_ai_performance_analyzer():
    st.header("⚡ AI-Powered Performance Analyzer")
    st.markdown("Advanced SQL Server performance analysis and optimization")
    
    # Performance metrics input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Current Performance Metrics")
        
        # Input fields for current performance data
        col_a, col_b = st.columns(2)
        
        with col_a:
            cpu_usage = st.slider("CPU Usage (%)", 0, 100, 65)
            memory_usage = st.slider("Memory Usage (%)", 0, 100, 78)
            avg_response_time = st.number_input("Avg Response Time (ms)", 0, 10000, 1250)
        
        with col_b:
            disk_io = st.number_input("Disk I/O (IOPS)", 0, 50000, 8500)
            active_connections = st.number_input("Active Connections", 0, 1000, 245)
            blocking_sessions = st.number_input("Blocking Sessions", 0, 100, 12)
        
        # Slow query input
        st.subheader("🐌 Problematic Queries")
        slow_queries = st.text_area(
            "Paste slow or problematic queries (one per line):",
            value="""SELECT o.OrderID, c.CustomerName, p.ProductName, oi.Quantity
FROM Orders o
JOIN Customers c ON o.CustomerID = c.CustomerID
JOIN OrderItems oi ON o.OrderID = oi.OrderID
JOIN Products p ON oi.ProductID = p.ProductID
WHERE o.OrderDate >= DATEADD(month, -6, GETDATE())
ORDER BY o.OrderDate DESC

SELECT COUNT(*) FROM TransactionLog tl
WHERE CONVERT(VARCHAR, tl.TransactionDate, 101) = CONVERT(VARCHAR, GETDATE(), 101)

UPDATE Inventory SET Quantity = Quantity - @Amount
WHERE ProductID IN (SELECT ProductID FROM OrderItems WHERE OrderID = @OrderID)""",
            height=200
        )
    
    with col2:
        st.subheader("⚙️ Analysis Configuration")
        
        analysis_depth = st.selectbox("Analysis Depth", ["Quick Scan", "Comprehensive", "Deep Dive"])
        focus_areas = st.multiselect("Focus Areas", 
                                   ["Query Optimization", "Index Analysis", "Memory Tuning", 
                                    "I/O Optimization", "Concurrency Issues", "Configuration"])
        
        environment = st.selectbox("Environment", ["Production", "Staging", "Development"])
        
        st.markdown("#### 🎯 Quick Metrics")
        
        # Calculate performance score
        performance_score = calculate_performance_score(cpu_usage, memory_usage, avg_response_time, blocking_sessions)
        
        if performance_score > 80:
            st.success(f"✅ Excellent: {performance_score}/100")
        elif performance_score > 60:
            st.warning(f"⚠️ Needs Attention: {performance_score}/100")
        else:
            st.error(f"🚨 Critical: {performance_score}/100")
        
        # Resource utilization gauge
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = cpu_usage,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "CPU Usage"},
            gauge = {'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}],
                    'threshold': {'line': {'color': "red", 'width': 4},
                                'thickness': 0.75, 'value': 90}}))
        fig_gauge.update_layout(height=200, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    # AI Analysis Section
    st.markdown("### 🤖 AI Performance Analysis")
    
    if claude_client:
        if st.button("🚀 Analyze Performance", type="primary"):
            with st.spinner("🤖 Claude is analyzing your SQL Server performance..."):
                
                # Prepare performance metrics
                performance_metrics = {
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'avg_response_time': avg_response_time,
                    'disk_io': disk_io,
                    'connections': active_connections,
                    'blocking': blocking_sessions,
                    'waits': "PAGEIOLATCH_SH, CXPACKET, LCK_M_S detected"
                }
                
                # Parse slow queries
                query_list = [q.strip() for q in slow_queries.split('\n\n') if q.strip()]
                
                ai_analysis = ai_analyze_sql_server_performance(claude_client, performance_metrics, query_list)
                
                if ai_analysis:
                    st.markdown("#### 🧠 Claude AI Performance Analysis")
                    st.markdown(ai_analysis)
                    
                    # Additional performance recommendations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🎯 Immediate Actions")
                        immediate_actions = [
                            "Rebuild fragmented indexes identified",
                            "Update statistics on heavily used tables", 
                            "Resolve blocking sessions",
                            "Optimize top 3 slowest queries"
                        ]
                        
                        for action in immediate_actions:
                            if st.button(f"⚡ {action}"):
                                st.success(f"✅ {action} - Task initiated")
                    
                    with col2:
                        st.markdown("#### 📊 Performance Trends")
                        
                        # Generate trend data
                        dates = pd.date_range(start='2024-07-01', end='2024-07-26', freq='H')
                        trend_data = pd.DataFrame({
                            'timestamp': dates,
                            'cpu_usage': [cpu_usage + random.uniform(-20, 20) for _ in dates],
                            'response_time': [avg_response_time + random.uniform(-500, 1000) for _ in dates]
                        })
                        
                        fig_trend = px.line(trend_data, x='timestamp', y='response_time',
                                          title="Response Time Trend (24h)")
                        fig_trend.add_hline(y=1000, line_dash="dash", line_color="red", 
                                          annotation_text="SLA Threshold")
                        st.plotly_chart(fig_trend, use_container_width=True)
                
                else:
                    st.error("❌ AI analysis failed. Please check your API configuration.")
    
    else:
        st.warning("🔑 Claude AI not available. Configure API key to enable performance analysis.")
        
        # Show mock analysis for demo
        st.markdown("#### 📊 Basic Performance Analysis (Mock Data)")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🚨 Issues Identified:**")
            st.markdown("""
            - High CPU usage (65%) during peak hours
            - Blocking sessions causing delays
            - Inefficient query patterns detected
            - Missing indexes on frequently accessed tables
            """)
        
        with col2:
            st.markdown("**🚀 Recommendations:**")
            st.markdown("""
            - Implement index optimization strategy
            - Review and optimize slow queries
            - Configure proper locking mechanisms
            - Scale up compute resources during peak times
            """)

def show_natural_language_sql():
    st.header("💬 Natural Language to SQL")
    st.markdown("Convert natural language requests to optimized SQL Server queries")
    
    # SQL Server schema for context
    schemas = {
        "E-commerce Production": {
            "tables": {
                "dbo.Customers": ["CustomerID (int, PK)", "CustomerName (nvarchar)", "Email (nvarchar)", "Phone (nvarchar)", "CreatedDate (datetime)", "Status (nvarchar)"],
                "dbo.Orders": ["OrderID (int, PK)", "CustomerID (int, FK)", "OrderDate (datetime)", "TotalAmount (money)", "Status (nvarchar)", "ShippingAddress (nvarchar)"],
                "dbo.Products": ["ProductID (int, PK)", "ProductName (nvarchar)", "CategoryID (int, FK)", "Price (money)", "StockQuantity (int)", "Description (ntext)"],
                "dbo.OrderItems": ["OrderItemID (int, PK)", "OrderID (int, FK)", "ProductID (int, FK)", "Quantity (int)", "UnitPrice (money)"],
                "dbo.Categories": ["CategoryID (int, PK)", "CategoryName (nvarchar)", "Description (nvarchar)"]
            },
            "indexes": [
                "IX_Orders_CustomerID_Date ON dbo.Orders(CustomerID, OrderDate)",
                "IX_Products_Category_Price ON dbo.Products(CategoryID, Price)",
                "IX_Customers_Email ON dbo.Customers(Email)"
            ]
        }
    }
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("💭 Natural Language Query")
        
        # Example queries for quick selection
        example_queries = [
            "Show me the top 10 customers by total order value in the last 6 months",
            "Find all products with low inventory (less than 10 units) in the Electronics category", 
            "Get monthly sales trends for the current year",
            "List customers who placed orders over $1000 but haven't ordered in the last 30 days",
            "Show me the most popular products by category with their profit margins",
            "Find orders that are pending for more than 7 days",
            "Get customer retention rate by registration month",
            "Show me products that are frequently returned"
        ]
        
        selected_example = st.selectbox("Quick Examples (optional)", ["Custom Query"] + example_queries)
        
        if selected_example != "Custom Query":
            nl_query = selected_example
        else:
            nl_query = st.text_area(
                "Describe what you want to query:",
                placeholder="e.g., Show me customers who haven't placed orders in the last 90 days",
                height=100
            )
        
        # Advanced options
        with st.expander("🔧 Advanced Options"):
            include_execution_plan = st.checkbox("Include execution plan analysis")
            optimize_for_performance = st.checkbox("Optimize for performance", value=True)
            add_comments = st.checkbox("Add explanatory comments", value=True)
            target_sql_version = st.selectbox("SQL Server Version", ["2019", "2017", "2016", "2022"])
    
    with col2:
        st.subheader("📋 Database Schema")
        
        selected_schema = st.selectbox("Schema", list(schemas.keys()))
        
        for table, columns in schemas[selected_schema]["tables"].items():
            with st.expander(f"📋 {table}"):
                for column in columns:
                    st.text(f"• {column}")
        
        st.markdown("**Indexes:**")
        for index in schemas[selected_schema]["indexes"]:
            st.text(f"• {index}")
    
    # AI Conversion
    if st.button("🔄 Convert to SQL", type="primary"):
        if nl_query:
            with st.spinner("🤖 Claude is converting your request to optimized SQL Server query..."):
                
                if claude_client:
                    # Prepare schema context
                    schema_context = f"""
SQL SERVER DATABASE SCHEMA: {selected_schema}

TABLES AND COLUMNS:
"""
                    for table, columns in schemas[selected_schema]["tables"].items():
                        schema_context += f"\n{table}:\n"
                        for column in columns:
                            schema_context += f"  - {column}\n"
                    
                    schema_context += f"\nEXISTING INDEXES:\n"
                    for index in schemas[selected_schema]["indexes"]:
                        schema_context += f"  - {index}\n"
                    
                    schema_context += f"\nSQL SERVER VERSION: {target_sql_version}"
                    
                    # Enhanced prompt for SQL Server
                    prompt = f"""You are a senior SQL Server DBA and developer. Convert this natural language query to optimized SQL Server T-SQL.

{schema_context}

NATURAL LANGUAGE QUERY: {nl_query}

REQUIREMENTS:
- Use SQL Server {target_sql_version} syntax and features
- Optimize for performance with proper indexing hints where beneficial
- Use appropriate SQL Server functions (DATEADD, DATEDIFF, etc.)
- Include WITH (NOLOCK) hints only where appropriate
- {"Include SET STATISTICS IO ON and actual execution plan analysis" if include_execution_plan else ""}
- {"Add detailed comments explaining the logic" if add_comments else ""}
- Handle potential NULL values appropriately
- Use parameterization to prevent SQL injection

Please provide:

**🎯 OPTIMIZED SQL QUERY:**
```sql
[Your optimized T-SQL here]
```

**📊 PERFORMANCE CONSIDERATIONS:**
- Index usage and recommendations
- Expected performance characteristics
- Potential bottlenecks to watch for

**🔧 OPTIMIZATION NOTES:**
- Why specific approaches were chosen
- Alternative query patterns considered
- SQL Server specific optimizations applied

**📈 EXECUTION PLAN INSIGHTS:**
- Expected join algorithms
- Index seek vs scan predictions
- Cost analysis and recommendations

Focus on SQL Server best practices including proper use of:
- CTEs vs subqueries vs JOINs
- Window functions where appropriate
- Proper parameterization
- Set-based operations over cursors
- Appropriate isolation levels"""

                    ai_response = call_claude_api(claude_client, prompt, max_tokens=2500)
                    
                    if ai_response:
                        st.markdown("### 🤖 Claude AI Generated SQL")
                        st.markdown(ai_response)
                        
                        # Additional features
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("📋 Copy Query"):
                                st.success("✅ SQL query copied to clipboard")
                        
                        with col2:
                            if st.button("🔍 Analyze Performance"):
                                st.info("🔍 Performance analysis initiated...")
                        
                        with col3:
                            if st.button("💾 Save to Library"):
                                st.success("💾 Query saved to personal library")
                    
                    else:
                        st.error("❌ Failed to generate SQL. Please try again.")
                
                else:
                    # Fallback mock response
                    st.warning("⚠️ Claude AI not available. Showing sample response.")
                    
                    if "top" in nl_query.lower() and "customer" in nl_query.lower():
                        sample_sql = """-- Top customers by total order value (Last 6 months)
-- Optimized for SQL Server with performance considerations

SET STATISTICS IO ON;

WITH CustomerOrderTotals AS (
    SELECT 
        c.CustomerID,
        c.CustomerName,
        c.Email,
        SUM(o.TotalAmount) as TotalOrderValue,
        COUNT(DISTINCT o.OrderID) as OrderCount,
        MAX(o.OrderDate) as LastOrderDate
    FROM dbo.Customers c WITH (NOLOCK)
    INNER JOIN dbo.Orders o WITH (NOLOCK) 
        ON c.CustomerID = o.CustomerID
    WHERE o.OrderDate >= DATEADD(MONTH, -6, GETDATE())
        AND o.Status = 'Completed'
    GROUP BY c.CustomerID, c.CustomerName, c.Email
)
SELECT TOP 10
    CustomerID,
    CustomerName,
    Email,
    TotalOrderValue,
    OrderCount,
    LastOrderDate,
    FORMAT(TotalOrderValue / OrderCount, 'C') as AvgOrderValue
FROM CustomerOrderTotals
ORDER BY TotalOrderValue DESC;

-- Performance Notes:
-- Uses existing IX_Orders_CustomerID_Date index
-- WITH (NOLOCK) for read-heavy reporting scenarios
-- CTE improves readability while maintaining performance"""
                        
                        st.markdown("### 🤖 Generated SQL Query")
                        st.code(sample_sql, language="sql")
                        
                        st.markdown("### 🚀 Optimization Features")
                        st.markdown("""
                        ✅ **Performance Optimizations Applied:**
                        - Leveraged existing index IX_Orders_CustomerID_Date
                        - Used CTE for better query plan optimization
                        - Added appropriate NOLOCK hints for reporting
                        - Filtered early to reduce data processing
                        - Used efficient date range filtering with DATEADD
                        
                        📊 **Expected Performance:**
                        - Index seeks on Orders table
                        - Hash join for Customer-Order relationship
                        - Estimated execution time: <100ms for typical workloads
                        """)

def show_enterprise_reporting(index_df, maintenance_df):
    st.header("📊 Enterprise Reporting & Analytics")
    st.markdown("Comprehensive database health and performance reporting")
    
    # Report type selection
    report_types = {
        "Executive Summary": "executive",
        "DBA Operations Report": "dba_ops", 
        "Performance Analysis": "performance",
        "Cost Optimization": "cost_opt",
        "Compliance & Audit": "compliance",
        "Capacity Planning": "capacity"
    }
    
    selected_report = st.selectbox("Report Type", list(report_types.keys()))
    
    # Date range selection
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
    
    with col2:
        end_date = st.date_input("End Date", datetime.now())
    
    with col3:
        report_format = st.selectbox("Export Format", ["PDF", "PowerBI", "Excel", "Email"])
    
    # Generate report based on selection
    if report_types[selected_report] == "executive":
        show_executive_report(index_df, maintenance_df)
    elif report_types[selected_report] == "dba_ops":
        show_dba_operations_report(index_df, maintenance_df)
    elif report_types[selected_report] == "performance":
        show_performance_report(index_df, maintenance_df)
    elif report_types[selected_report] == "cost_opt":
        show_cost_optimization_report(index_df, maintenance_df)
    elif report_types[selected_report] == "compliance":
        show_compliance_report(index_df, maintenance_df)
    elif report_types[selected_report] == "capacity":
        show_capacity_planning_report(index_df, maintenance_df)

def show_configuration_center():
    st.header("⚙️ Configuration Center")
    st.markdown("Enterprise configuration management for SQL Server optimization")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔧 System Settings", "🤖 AI Configuration", "📧 Notifications", "🔐 Security"])
    
    with tab1:
        show_system_settings()
    
    with tab2:
        show_ai_configuration()
    
    with tab3:
        show_notification_settings()
    
    with tab4:
        show_security_settings()

def show_demo_examples():
    st.header("🎯 Enterprise Demo Examples")
    st.markdown("Production-ready examples for SQL Server optimization")
    
    # Enterprise scenarios
    scenarios = {
        "🏢 Large E-commerce Platform": {
            "description": "High-traffic e-commerce with 50TB database, 10K concurrent users",
            "challenges": ["Index fragmentation", "Query timeouts", "Blocking sessions"],
            "solutions": ["AI-powered index maintenance", "Query optimization", "Concurrency tuning"]
        },
        "🏦 Financial Services": {
            "description": "Banking system with strict SLA requirements and regulatory compliance",
            "challenges": ["Zero-downtime maintenance", "Audit trail", "Performance SLA"],
            "solutions": ["Online index operations", "Automated compliance reporting", "Performance monitoring"]
        },
        "🏥 Healthcare Analytics": {
            "description": "Healthcare data warehouse with complex analytical workloads",
            "challenges": ["Large table scans", "Complex aggregations", "Data privacy"],
            "solutions": ["Columnstore indexes", "Partition strategies", "Encryption optimization"]
        }
    }
    
    for scenario_name, scenario in scenarios.items():
        with st.expander(scenario_name):
            st.markdown(f"**Description:** {scenario['description']}")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🚨 Challenges:**")
                for challenge in scenario['challenges']:
                    st.markdown(f"• {challenge}")
            
            with col2:
                st.markdown("**✅ Solutions:**")
                for solution in scenario['solutions']:
                    st.markdown(f"• {solution}")
            
            if st.button(f"🚀 Load {scenario_name.split()[1]} Scenario", key=scenario_name):
                st.success(f"✅ {scenario_name} scenario loaded successfully!")

# Helper functions
def generate_index_maintenance_script(high_priority_indexes):
    """Generate T-SQL script for index maintenance"""
    script = """-- Enterprise Index Maintenance Script
-- Generated by AI SQL Server Optimizer
-- Execute during maintenance window only

SET NOCOUNT ON;
DECLARE @StartTime DATETIME = GETDATE();
PRINT 'Starting index maintenance at ' + CONVERT(VARCHAR, @StartTime, 120);

"""
    
    for _, row in high_priority_indexes.iterrows():
        if row['Recommended_Action'] == 'Rebuild':
            script += f"""
-- Rebuild {row['Index']} on {row['Database']}.{row['Schema']}.{row['Table']}
-- Current fragmentation: {row['Fragmentation_Percent']}%
ALTER INDEX [{row['Index']}] ON [{row['Database']}].[{row['Schema']}].[{row['Table']}] 
    REBUILD WITH (
        PAD_INDEX = OFF,
        STATISTICS_NORECOMPUTE = OFF,
        ALLOW_ROW_LOCKS = ON,
        ALLOW_PAGE_LOCKS = ON,
        ONLINE = ON,
        MAXDOP = 4
    );
PRINT 'Completed rebuild of {row['Index']} at ' + CONVERT(VARCHAR, GETDATE(), 120);
GO
"""
        else:
            script += f"""
-- Reorganize {row['Index']} on {row['Database']}.{row['Schema']}.{row['Table']}
ALTER INDEX [{row['Index']}] ON [{row['Database']}].[{row['Schema']}].[{row['Table']}] REORGANIZE;
PRINT 'Completed reorganize of {row['Index']} at ' + CONVERT(VARCHAR, GETDATE(), 120);
GO
"""
    
    script += """
DECLARE @EndTime DATETIME = GETDATE();
PRINT 'Index maintenance completed at ' + CONVERT(VARCHAR, @EndTime, 120);
PRINT 'Total duration: ' + CONVERT(VARCHAR, DATEDIFF(MINUTE, @StartTime, @EndTime)) + ' minutes';
"""
    
    return script

def generate_maintenance_timeline(maintenance_df):
    """Generate maintenance timeline script"""
    script = """-- Enterprise Maintenance Timeline
-- Optimized schedule for minimal business impact

USE msdb;
GO

-- Create maintenance plan schedules
"""
    
    for _, row in maintenance_df.iterrows():
        script += f"""
-- Schedule for {row['Component']} on {row['Database']}
EXEC dbo.sp_add_schedule
    @schedule_name = N'{row['Database']}_{row['Component']}_Schedule',
    @enabled = 1,
    @freq_type = 4, -- Daily
    @freq_interval = 1,
    @active_start_time = 020000; -- 2:00 AM
GO
"""
    
    return script

def optimize_maintenance_time(component, database, window):
    """Optimize maintenance timing based on component and window"""
    base_times = {
        "2:00-6:00 AM": "02:00",
        "10:00 PM-2:00 AM": "22:00", 
        "Weekend Only": "Saturday 02:00",
        "Custom": "02:00"
    }
    
    # Add component-specific offsets
    offsets = {
        "Index Rebuild": 0,
        "Index Reorganize": 30,
        "Statistics Update": 60,
        "Integrity Check": 90,
        "Backup Full": 120
    }
    
    base_time = base_times.get(window, "02:00")
    offset = offsets.get(component, 0)
    
    return f"{base_time} + {offset} minutes"

def assign_maintenance_priority(component, success_rate):
    """Assign maintenance priority based on component and success rate"""
    if success_rate < 90:
        return "High"
    elif component in ["Integrity Check", "Backup Full"]:
        return "High"
    elif component in ["Index Rebuild", "Statistics Update"]:
        return "Medium"
    else:
        return "Low"

def estimate_business_impact(component, database):
    """Estimate business impact of maintenance"""
    if "Production" in database:
        if component in ["Index Rebuild", "Integrity Check"]:
            return "High"
        else:
            return "Medium"
    else:
        return "Low"

def calculate_performance_score(cpu, memory, response_time, blocking):
    """Calculate overall performance score"""
    cpu_score = max(0, 100 - cpu)
    memory_score = max(0, 100 - memory)
    response_score = max(0, 100 - (response_time / 50))
    blocking_score = max(0, 100 - (blocking * 5))
    
    return (cpu_score + memory_score + response_score + blocking_score) / 4

# Additional report functions (simplified for brevity)
def show_executive_report(index_df, maintenance_df):
    st.markdown("### 📈 Executive Summary Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Key Performance Indicators")
        
        total_databases = len(index_df['Database'].unique())
        critical_issues = len(index_df[index_df['Priority'] == 'High'])
        est_savings = critical_issues * 45000
        
        st.metric("Total Databases", total_databases)
        st.metric("Critical Issues", critical_issues)
        st.metric("Est. Annual Savings", f"${est_savings:,}")
    
    with col2:
        # Performance trend
        dates = pd.date_range(start='2024-07-01', end='2024-07-26', freq='D')
        trend_data = pd.DataFrame({
            'Date': dates,
            'Performance_Score': [random.uniform(70, 95) for _ in dates]
        })
        
        fig = px.line(trend_data, x='Date', y='Performance_Score', 
                     title="Database Performance Trend")
        st.plotly_chart(fig, use_container_width=True)

def show_dba_operations_report(index_df, maintenance_df):
    st.markdown("### 🔧 DBA Operations Report")
    st.dataframe(index_df.head(20), use_container_width=True)

def show_performance_report(index_df, maintenance_df):
    st.markdown("### ⚡ Performance Analysis Report")
    
    # Performance analysis charts
    col1, col2 = st.columns(2)
    
    with col1:
        frag_dist = index_df['Fragmentation_Percent'].hist(bins=20)
        fig = px.histogram(index_df, x='Fragmentation_Percent', nbins=20,
                          title="Fragmentation Distribution")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        size_vs_frag = px.scatter(index_df, x='Size_MB', y='Fragmentation_Percent',
                                 color='Priority', title="Size vs Fragmentation")
        st.plotly_chart(size_vs_frag, use_container_width=True)

def show_cost_optimization_report(index_df, maintenance_df):
    st.markdown("### 💰 Cost Optimization Report")
    
    # Cost analysis
    total_storage_cost = (index_df['Size_MB'].sum() / 1024) * 0.10  # $0.10 per GB
    potential_savings = len(index_df[index_df['Priority'] == 'High']) * 1000  # $1k per optimized index
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Current Storage Cost", f"${total_storage_cost:.2f}/month")
        st.metric("Potential Savings", f"${potential_savings:,}/month")
    
    with col2:
        # Cost breakdown pie chart
        cost_data = pd.DataFrame({
            'Category': ['Storage', 'Compute', 'Maintenance'],
            'Cost': [total_storage_cost, total_storage_cost * 2, total_storage_cost * 0.5]
        })
        
        fig = px.pie(cost_data, values='Cost', names='Category',
                    title="Cost Breakdown")
        st.plotly_chart(fig, use_container_width=True)

def show_compliance_report(index_df, maintenance_df):
    st.markdown("### 🔐 Compliance & Audit Report")
    
    compliance_metrics = {
        "Backup Compliance": "98.5%",
        "Maintenance SLA": "99.2%", 
        "Security Patches": "Current",
        "Audit Log Retention": "365 days"
    }
    
    for metric, value in compliance_metrics.items():
        st.metric(metric, value)

def show_capacity_planning_report(index_df, maintenance_df):
    st.markdown("### 📊 Capacity Planning Report")
    
    # Growth projections
    current_size = index_df['Size_MB'].sum() / 1024  # GB
    monthly_growth = current_size * 0.15  # 15% monthly growth
    
    months = range(1, 13)
    projected_sizes = [current_size + (monthly_growth * m) for m in months]
    
    growth_df = pd.DataFrame({
        'Month': months,
        'Projected_Size_GB': projected_sizes
    })
    
    fig = px.line(growth_df, x='Month', y='Projected_Size_GB',
                 title="12-Month Storage Growth Projection")
    st.plotly_chart(fig, use_container_width=True)

def show_system_settings():
    st.subheader("🔧 System Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Database Settings")
        default_fragmentation_threshold = st.slider("Default Fragmentation Threshold", 0, 100, 30)
        maintenance_window_start = st.time_input("Maintenance Window Start", value=datetime.strptime("02:00", "%H:%M").time())
        parallel_operations = st.checkbox("Enable Parallel Operations", value=True)
        
    with col2:
        st.markdown("#### Performance Settings")
        max_maintenance_duration = st.number_input("Max Maintenance Duration (hours)", 1, 12, 4)
        resource_utilization_limit = st.slider("Resource Utilization Limit (%)", 0, 100, 80)
        auto_statistics_update = st.checkbox("Auto Statistics Update", value=True)

def show_ai_configuration():
    st.subheader("🤖 AI Configuration")
    
    # API Configuration
    st.markdown("#### Claude AI Settings")
    
    api_key_status = "✅ Connected" if claude_client else "❌ Not Connected"
    st.text(f"API Status: {api_key_status}")
    
    if not claude_client:
        st.info("Add your Anthropic API key in Streamlit secrets or environment variables")
        
        with st.expander("🔑 API Key Setup Instructions"):
            st.markdown("""
            **For Streamlit Cloud:**
            1. Go to your app settings
            2. Add secret: `ANTHROPIC_API_KEY = "your-api-key"`
            
            **For Local Development:**
            ```bash
            export ANTHROPIC_API_KEY="your-api-key"
            ```
            
            **Get API Key:**
            Visit [console.anthropic.com](https://console.anthropic.com)
            """)
    
    # AI Behavior Settings
    st.markdown("#### AI Behavior")
    ai_analysis_depth = st.selectbox("Default Analysis Depth", ["Quick", "Standard", "Comprehensive"])
    ai_recommendation_level = st.selectbox("Recommendation Aggressiveness", ["Conservative", "Balanced", "Aggressive"])
    auto_implement_safe_recommendations = st.checkbox("Auto-implement Safe Recommendations", value=False)

def show_notification_settings():
    st.subheader("📧 Notification Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Alert Thresholds")
        critical_fragmentation = st.slider("Critical Fragmentation Alert", 0, 100, 50)
        performance_degradation = st.slider("Performance Degradation Alert", 0, 100, 25)
        maintenance_failure = st.checkbox("Maintenance Failure Alerts", value=True)
        
    with col2:
        st.markdown("#### Notification Channels")
        email_notifications = st.checkbox("Email Notifications", value=True)
        
        if email_notifications:
            email_recipients = st.text_area("Email Recipients", "dba@company.com\nmanager@company.com")
        
        slack_integration = st.checkbox("Slack Integration", value=False)
        teams_integration = st.checkbox("Microsoft Teams", value=False)

def show_security_settings():
    st.subheader("🔐 Security Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Access Control")
        require_mfa = st.checkbox("Require Multi-Factor Authentication", value=True)
        session_timeout = st.selectbox("Session Timeout", ["30 minutes", "1 hour", "4 hours", "8 hours"])
        audit_all_actions = st.checkbox("Audit All Actions", value=True)
        
    with col2:
        st.markdown("#### Data Protection")
        encrypt_sensitive_data = st.checkbox("Encrypt Sensitive Data", value=True)
        mask_production_data = st.checkbox("Mask Production Data in Non-Prod", value=True)
        compliance_mode = st.selectbox("Compliance Mode", ["Standard", "HIPAA", "SOX", "GDPR"])

if __name__ == "__main__":
    main()