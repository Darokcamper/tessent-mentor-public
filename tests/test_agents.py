import unittest
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from core.llm import llm
from agents.router_agent import route_question
from agents.scan_agent import ask_scan
from agents.sta_agent import ask_sta
from agents.planner_agent import generate_study_plan

class TestVLSIAgents(unittest.TestCase):
    
    def test_llm_connection(self):
        """Test that the LLM is configured and responds."""
        try:
            response = llm.invoke("Hi")
            self.assertIsNotNone(response.content)
            self.assertTrue(len(response.content) > 0)
        except Exception as e:
            self.fail(f"LLM connection failed: {e}")
            
    def test_router_agent(self):
        """Test that the router correctly classifies simple topics."""
        topic_scan = route_question("Why are lockup latches needed in scan chains?")
        self.assertEqual(topic_scan, "SCAN")
        
        topic_sta = route_question("What is clock skew and setup time violation?")
        self.assertEqual(topic_sta, "STA")
        
    def test_expert_agent_fallback(self):
        """Test expert agent execution and response grounding."""
        # Query scan expert
        answer = ask_scan("Explain shift mode.")
        self.assertIsNotNone(answer)
        self.assertTrue(len(answer) > 10)
        
    def test_study_planner(self):
        """Test study planner generation logic with sample input."""
        asked = ["Explain scan stitching.", "What is March C-?"]
        evals = ["Score: 8\nStrong understanding of stitching.", "Score: 4\nConfused March C- with March A."]
        plan = generate_study_plan(asked, evals)
        self.assertIsNotNone(plan)
        self.assertIn("Strengths", plan)
        self.assertIn("Weak", plan)

if __name__ == "__main__":
    unittest.main()
