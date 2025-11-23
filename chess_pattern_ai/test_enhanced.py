#!/usr/bin/env python3
"""
Enhanced Pattern Learning Test Script

Tests both backward compatibility and enhanced features
"""

import sys
import os

def test_import():
    """Test that enhanced modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from enhanced_headless_trainer import CompatibleHeadlessTrainer
        print("   ✓ Enhanced trainer imports successfully")
        return True
    except ImportError as e:
        print(f"   ❌ Enhanced trainer import failed: {e}")
        return False

def test_original_mode():
    """Test original mode compatibility"""
    print("\n🧪 Testing original mode compatibility...")
    
    try:
        from enhanced_headless_trainer import CompatibleHeadlessTrainer
        
        trainer = CompatibleHeadlessTrainer(enhanced_mode=False)
        print("   ✓ Original mode trainer created")
        
        # Quick API test
        stats = trainer.prioritizer.get_statistics()
        print(f"   ✓ Statistics API works: {stats['patterns_learned']} patterns")
        
        trainer.close()
        print("   ✓ Original mode test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Original mode test failed: {e}")
        return False

def test_enhanced_mode():
    """Test enhanced mode functionality"""
    print("\n🧪 Testing enhanced mode functionality...")
    
    try:
        from enhanced_headless_trainer import CompatibleHeadlessTrainer
        
        trainer = CompatibleHeadlessTrainer(enhanced_mode=True)
        print("   ✓ Enhanced mode trainer created")
        
        # Check if enhanced features are available
        if hasattr(trainer.prioritizer, 'enhanced_mode'):
            print(f"   ✓ Enhanced mode: {trainer.prioritizer.enhanced_mode}")
        
        trainer.close()
        print("   ✓ Enhanced mode test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced mode test failed: {e}")
        return False

def test_quick_training():
    """Test quick training session"""
    print("\n🧪 Testing quick training session...")
    
    try:
        from enhanced_headless_trainer import CompatibleHeadlessTrainer
        
        # Test both modes
        for enhanced in [False, True]:
            mode_name = "Enhanced" if enhanced else "Original"
            print(f"   Testing {mode_name} mode training...")
            
            trainer = CompatibleHeadlessTrainer(
                db_path=f'test_{mode_name.lower()}.db',
                enhanced_mode=enhanced
            )
            
            # Train for just 2 games
            trainer.train(num_games=2, verbose=False, progress_interval=1)
            
            print(f"   ✓ {mode_name} mode: {trainer.wins}W-{trainer.losses}L-{trainer.draws}D")
            trainer.close()
        
        print("   ✓ Quick training test passed")
        return True
        
    except Exception as e:
        print(f"   ❌ Quick training test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 ENHANCED PATTERN LEARNING INTEGRATION TESTS")
    print("=" * 55)
    
    tests = [
        test_import,
        test_original_mode,
        test_enhanced_mode,
        test_quick_training
    ]
    
    results = []
    for test_func in tests:
        result = test_func()
        results.append(result)
    
    print("\n" + "=" * 55)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"🎉 ALL TESTS PASSED! ({passed}/{total})")
        print("\n✅ Integration is working correctly!")
        print("\n🚀 Next steps:")
        print("   python enhanced_headless_trainer.py 10          # Test original mode")
        print("   python enhanced_headless_trainer.py 10 --enhanced  # Test enhanced mode")
        return True
    else:
        print(f"⚠️ SOME TESTS FAILED ({passed}/{total})")
        print("\nCheck the error messages above for details.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
