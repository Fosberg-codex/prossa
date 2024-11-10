from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime
import uuid
import logging
from pathlib import Path

from .llm_manager import LLMManager, ModelType
from .validator import ConfidenceValidator, ValidationType
from .embeddings import EmbeddingManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Agent:
    def __init__(self, persist_directory: Optional[str] = "./prossa_data"):
        """Initialize the Prossa Agent with its components"""
        self.logger = logging.getLogger(__name__)
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.llm_manager = LLMManager()
        self.validator = ConfidenceValidator()
        self.embedding_manager = EmbeddingManager(
            persist_directory=str(self.persist_directory / "embeddings")
        )
        
        self.logger.info("Prossa Agent initialized successfully")
    
    async def analyze_dataset(self, 
                            dataset: pd.DataFrame,
                            dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze dataset and generate preprocessing recommendations
        Returns a comprehensive analysis report
        """
        dataset_id = dataset_name or f"dataset_{uuid.uuid4().hex[:8]}"
        dataset_metadata = self._extract_dataset_metadata(dataset)
        
        self.logger.info(f"Starting analysis for dataset: {dataset_id}")
        
        # Initial screening with lightweight model
        initial_analysis = await self._perform_initial_screening(
            dataset,
            dataset_metadata
        )
        
        # Comprehensive analysis based on initial screening
        analysis_results = await self._perform_comprehensive_analysis(
            dataset,
            initial_analysis,
            dataset_metadata
        )
        
        # Generate and validate recommendations
        recommendations = await self._generate_recommendations(
            analysis_results,
            dataset_metadata
        )
        
        # Store validated recommendations
        self._store_recommendations(recommendations, dataset_id, dataset_metadata)
        
        return self._generate_report(recommendations, dataset_metadata)
    
    async def _perform_initial_screening(self,
                                      dataset: pd.DataFrame,
                                      metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Perform quick initial screening of dataset"""
        prompt = self._create_screening_prompt(dataset, metadata)
        
        response = await self.llm_manager.process_task(
            task_type=ModelType.INITIAL_SCREENING,
            prompt=prompt,
            dataset_complexity=metadata["complexity"]
        )
        
        return response
    
    async def _perform_comprehensive_analysis(self,
                                           dataset: pd.DataFrame,
                                           initial_analysis: Dict[str, Any],
                                           metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed analysis based on initial screening"""
        analysis_prompt = self._create_analysis_prompt(
            dataset,
            initial_analysis,
            metadata
        )
        
        response = await self.llm_manager.process_task(
            task_type=ModelType.ANALYSIS,
            prompt=analysis_prompt,
            dataset_complexity=metadata["complexity"]
        )
        
        return response
    
    async def _generate_recommendations(self,
                                     analysis: Dict[str, Any],
                                     metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate and validate preprocessing recommendations"""
        recommendations = []
        
        for task_type in ValidationType:
            if self._should_process_task(task_type, analysis):
                recommendation = await self._generate_task_recommendation(
                    task_type,
                    analysis,
                    metadata
                )
                
                # Validate recommendation
                validation_result = self.validator.validate_recommendation(
                    recommendation,
                    task_type,
                    metadata
                )
                
                if validation_result["passed"]:
                    recommendations.append({
                        "type": task_type.value,
                        "recommendation": recommendation,
                        "validation": validation_result
                    })
                else:
                    self.logger.warning(
                        f"Recommendation for {task_type.value} failed validation"
                    )
        
        return recommendations
    
    def _extract_dataset_metadata(self, dataset: pd.DataFrame) -> Dict[str, Any]:
        """Extract metadata from dataset for analysis"""
        return {
            "type": self._determine_dataset_type(dataset),
            "shape": dataset.shape,
            "columns": list(dataset.columns),
            "dtypes": dataset.dtypes.astype(str).to_dict(),
            "complexity": self._calculate_complexity(dataset),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _determine_dataset_type(self, dataset: pd.DataFrame) -> str:
        """Determine the type of dataset based on its characteristics"""
        # Implementation based on dataset characteristics
        return "tabular"  # For now, we only handle tabular data
    
    def _calculate_complexity(self, dataset: pd.DataFrame) -> float:
        """Calculate dataset complexity score"""
        factors = [
            len(dataset.columns) / 100,  # Normalized number of features
            len(dataset) / 10000,  # Normalized number of samples
            dataset.dtypes.nunique() / len(dataset.dtypes),  # Data type diversity
            dataset.isnull().mean().mean(),  # Missing value ratio
            len([col for col in dataset.columns 
                 if dataset[col].dtype == 'object']) / len(dataset.columns)  # Categorical ratio
        ]
        
        # Weighted average of complexity factors
        weights = [0.3, 0.2, 0.2, 0.15, 0.15]
        complexity = np.average(factors, weights=weights)
        
        return min(max(complexity, 0.0), 1.0)
    
    def _store_recommendations(self,
                             recommendations: List[Dict[str, Any]],
                             dataset_id: str,
                             metadata: Dict[str, Any]) -> None:
        """Store recommendations in vector database"""
        for rec in recommendations:
            self.embedding_manager.store_recommendation(
                content=str(rec["recommendation"]),
                metadata={
                    "dataset_id": dataset_id,
                    "type": rec["type"],
                    **metadata
                },
                id=f"{dataset_id}_{rec['type']}_{uuid.uuid4().hex[:8]}"
            )
    
    def _generate_report(self,
                        recommendations: List[Dict[str, Any]],
                        metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed report of recommendations"""
        return {
            "dataset_metadata": metadata,
            "recommendations": recommendations,
            "summary": self._create_summary(recommendations),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _create_summary(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create a summary of all recommendations"""
        return {
            "total_recommendations": len(recommendations),
            "types": [rec["type"] for rec in recommendations],
            "average_confidence": np.mean([
                rec["validation"]["confidence_score"] 
                for rec in recommendations
            ])
        }
    
    def _should_process_task(self,
                           task_type: ValidationType,
                           analysis: Dict[str, Any]) -> bool:
        """Determine if a specific task type should be processed"""
        # Implementation based on analysis results
        return True  # For now, process all task types
    
    async def _generate_task_recommendation(self,
                                         task_type: ValidationType,
                                         analysis: Dict[str, Any],
                                         metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendation for specific task type"""
        prompt = self._create_task_prompt(task_type, analysis, metadata)
        
        response = await self.llm_manager.process_task(
            task_type=ModelType[task_type.name],
            prompt=prompt,
            dataset_complexity=metadata["complexity"]
        )
        
        return {
            "id": uuid.uuid4().hex,
            "type": task_type.value,
            "content": response["content"],
            "model": response["model"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _create_screening_prompt(self,
                               dataset: pd.DataFrame,
                               metadata: Dict[str, Any]) -> str:
        """Create prompt for initial dataset screening"""
        return f"""Analyze this dataset for preprocessing needs:
        Columns: {metadata['columns']}
        Data Types: {metadata['dtypes']}
        Shape: {metadata['shape']}
        
        Provide a quick assessment of:
        1. Data quality issues
        2. Required preprocessing steps
        3. Potential challenges
        """
    
    def _create_analysis_prompt(self,
                              dataset: pd.DataFrame,
                              initial_analysis: Dict[str, Any],
                              metadata: Dict[str, Any]) -> str:
        """Create prompt for comprehensive analysis"""
        return f"""Based on initial screening results:
        {initial_analysis['content']}
        
        Perform detailed analysis for:
        1. Statistical properties
        2. Data distributions
        3. Feature relationships
        4. Quality metrics
        
        Dataset metadata:
        {metadata}
        """
    
    def _create_task_prompt(self,
                          task_type: ValidationType,
                          analysis: Dict[str, Any],
                          metadata: Dict[str, Any]) -> str:
        """Create prompt for specific task type"""
        return f"""Generate preprocessing recommendations for {task_type.value}:
        
        Analysis results:
        {analysis['content']}
        
        Dataset characteristics:
        {metadata}
        
        Provide specific, actionable recommendations with:
        1. Justification
        2. Expected impact
        3. Implementation details
        """ 