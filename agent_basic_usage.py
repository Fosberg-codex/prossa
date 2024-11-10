import pandas as pd
import asyncio
from prossa_agent.core.agent import Agent
from prossa_agent.utils.reporting import ReportGenerator

async def main():
    # Create sample dataset
    data = {
        'age': [25, 30, 35, None, 45, 50, 1000],
        'income': [30000, 45000, None, 55000, 65000, 75000, 85000],
        'education': ['High School', 'Bachelor', 'Master', 'PhD', 'Bachelor', None, 'High School'],
        'employed': ['Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes']
    }
    df = pd.DataFrame(data)
    
    # Initialize Prossa Agent
    agent = Agent()
    
    # Initialize Report Generator
    report_generator = ReportGenerator()
    
    try:
        # Analyze dataset
        print("Analyzing dataset...")
        analysis_results = await agent.analyze_dataset(
            dataset=df,
            dataset_name="sample_demographic_data"
        )
        
        # Generate reports in different formats
        print("\nGenerating reports...")
        
        # Console report
        report_generator.generate_report(
            analysis_results=analysis_results,
            format="console"
        )
        
        # JSON report
        report_generator.export_report(
            results=analysis_results,
            format="json",
            filename="sample_analysis_report.json"
        )
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main()) 