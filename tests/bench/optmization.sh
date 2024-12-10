# 中间代码优化测试
python3 ./tests/bench/evaluate.py --test-llvm --opt-level 1 --testcase-dir ./tests/testcase/optimize_test/
# 注：此处只检测生成的中间代码是否正确，未检测优化效果，优化结果会人工检测
# 注：请注意手动检测开启优化后，基础/进阶测试用例是否有错误: python3 ./tests/bench/evaluate.py --test-llvm --opt-level 1 --testcase-dir ./tests/testcase/functional_test/