"""Tests for A/B testing framework."""
import pytest
import tempfile
import os

from src.ab_testing import ABTestingFramework, ABTestConfig


@pytest.fixture
def temp_ab_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def framework(temp_ab_dir):
    return ABTestingFramework(data_dir=temp_ab_dir)


class TestABTestingFramework:
    def test_create_test(self, framework):
        test_id = framework.create_test(
            test_name="Strategy Test",
            strategies=["A", "B", "C"]
        )
        
        assert test_id is not None
        assert test_id.startswith("test_")
    
    def test_get_active_test(self, framework):
        assert framework.get_active_test() is None
        
        framework.create_test("Test 1", ["A", "B"])
        active = framework.get_active_test()
        
        assert active is not None
        assert active.test_name == "Test 1"
    
    def test_assign_strategy_deterministic(self, framework):
        framework.create_test("Det Test", ["X", "Y"], use_thompson_sampling=False)
        
        # Same borrower should always get same strategy
        s1 = framework.assign_strategy("BRW_001")
        s2 = framework.assign_strategy("BRW_001")
        s3 = framework.assign_strategy("BRW_001")
        
        assert s1 == s2 == s3
        assert s1 in ["X", "Y"]
    
    def test_assign_strategy_covers_all(self, framework):
        framework.create_test("Coverage", ["Alpha", "Beta"], use_thompson_sampling=False)
        
        strategies = set()
        for i in range(100):
            s = framework.assign_strategy(f"BRW_{i}")
            strategies.add(s)
        
        # Both strategies should be assigned to someone
        assert strategies == {"Alpha", "Beta"}
    
    def test_record_outcome(self, framework):
        test_id = framework.create_test("Outcome Test", ["S1", "S2"])
        
        framework.record_outcome(
            test_id=test_id,
            borrower_id="B1",
            strategy="S1",
            outcome="Fully Recovered",
            recovery_amount=5000
        )
        
        results = framework.get_test_results(test_id)
        assert "S1" in results
        assert results["S1"].total_assigned == 1
        assert results["S1"].fully_recovered == 1
    
    def test_calculate_significance_insufficient_data(self, framework):
        test_id = framework.create_test("Sig Test", ["A", "B"])
        
        result = framework.calculate_significance(test_id)
        assert result['significant'] == False
        assert 'Insufficient' in result['message']
    
    def test_calculate_significance_with_data(self, framework):
        test_id = framework.create_test("Sig Test 2", ["Good", "Bad"])
        
        # Add outcomes - Good strategy has 80% success, Bad has 20%
        for i in range(50):
            recovered = i < 40  # 80% recovery
            framework.record_outcome(test_id, f"G{i}", "Good", 
                "Fully Recovered" if recovered else "Written Off")
        
        for i in range(50):
            recovered = i < 10  # 20% recovery
            framework.record_outcome(test_id, f"B{i}", "Bad",
                "Fully Recovered" if recovered else "Written Off")
        
        result = framework.calculate_significance(test_id)
        assert result['p_value'] is not None
        assert result['p_value'] < 0.05  # Should be significant
        assert result['significant'] == True
    
    def test_get_best_strategy(self, framework):
        test_id = framework.create_test("Best Test", ["Winner", "Loser"])
        
        # Winner has 90% success
        for i in range(10):
            framework.record_outcome(test_id, f"W{i}", "Winner",
                "Fully Recovered" if i < 9 else "Written Off")
        
        # Loser has 10% success
        for i in range(10):
            framework.record_outcome(test_id, f"L{i}", "Loser",
                "Fully Recovered" if i < 1 else "Written Off")
        
        best = framework.get_best_strategy(test_id)
        assert best['strategy'] == "Winner"
        assert best['recovery_rate'] == pytest.approx(0.9)
    
    def test_thompson_sampling_assignment(self, framework):
        framework.create_test("TS Test", ["A", "B"], use_thompson_sampling=True)
        
        # Should still work (random sampling)
        strategies = [framework.assign_strategy(f"BRW_{i}") for i in range(20)]
        assert all(s in ["A", "B"] for s in strategies)
