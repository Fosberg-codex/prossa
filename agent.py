import asyncio
import pandas as pd
from pathlib import Path
from prossa_agent.core.prossa_agent import ProsaAgent
from prossa_agent.utils.config import SystemConfig, load_config

async def main():
    # Load dataset
    df = pd.read_csv(Path(__file__).parent / 'dataset' / 'military.csv')
    
    # Load configuration (this will load from .env)
    api_config, system_config = load_config()
    
    # Print to verify API key is loaded
    print(f"Gemini API Key found: {bool(api_config.gemini_api_key)}")
    
    # Initialize agent with loaded config
    agent = ProsaAgent(system_config)
    
    # Process dataset
    result = await agent.process_dataset(
        dataset=df,
        task_description="Prepare this dataset for machine learning"
    )
    
    if result.success:
        print(f"Processing successful! Confidence: {result.confidence}")
        processed_df = result.data
        processed_df.to_csv('processed_dataset.csv', index=False)
    else:
        print(f"Processing failed: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())