#!/usr/bin/env python3

import argparse
import difflib
import os
import shutil
import subprocess
import re
import datetime
import csv
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ANSI color codes for colored output
class Colors:
    RESET = "\033[0m"
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    BOLD = "\033[1m"

@dataclass
class Config:
    timeout: int = 600
    opt_level: int = 0
    output_dir: str = "./output"
    testcase_dir: str = "./tests/testcase"
    runtime_lib_dir: str = "./tests/sysy-runtime-lib"
    executable_path: str = "./target/release/nkucc"
    no_compile: bool = False
    no_test: bool = False
    test_llvm: bool = False

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Test automation script for compiler.")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout for each test case in seconds.")
    parser.add_argument("--opt-level", type=int, default=0, help="Optimization level for the compiler.")
    parser.add_argument("--output-dir", default="./output", help="Directory to store outputs.")
    parser.add_argument("--testcase-dir", default="./tests/testcase", help="Directory containing test cases.")
    parser.add_argument("--runtime-lib-dir", default="./tests/sysy-runtime-lib", help="Directory for runtime libraries.")
    parser.add_argument("--executable-path", default="./target/release/nkucc", help="Path to the compiler executable.")
    parser.add_argument("--no-compile", action="store_true", help="Skip the compilation step.")
    parser.add_argument("--no-test", action="store_true", help="Skip the testing step.")
    parser.add_argument("--test-llvm", action="store_true", help="Test llvm-ir generation.")

    args = parser.parse_args()

    return Config(
        timeout=args.timeout,
        opt_level=args.opt_level,
        output_dir=args.output_dir,
        testcase_dir=args.testcase_dir,
        runtime_lib_dir=args.runtime_lib_dir,
        executable_path=args.executable_path,
        no_compile=args.no_compile,
        no_test=args.no_test,
        test_llvm=args.test_llvm,
    )

def execute_command(command: str, timeout: int) -> Dict[str, Any]:
    """
    Execute a shell command with a timeout and capture its output.
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            timeout=timeout,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": "TIMEOUT",
        }
    except Exception as e:
        return {
            "returncode": None,
            "stdout": "",
            "stderr": str(e),
        }

def check_file(file1: str, file2: str, diff_file: Optional[str] = None) -> bool:
    """
    Compare two files. If they differ, and contain floating-point numbers, allow a tolerance.
    """
    try:
        with open(file1, "r") as f1, open(file2, "r") as f2:
            lines1 = [line.strip() for line in f1]
            lines2 = [line.strip() for line in f2]

        diff = list(difflib.unified_diff(lines1, lines2, fromfile=file1, tofile=file2))

        if diff_file and diff:
            with open(diff_file, "w") as df:
                df.writelines(line + '\n' for line in diff)

        if not diff:
            return True

        # Compare floating-point numbers with a tolerance
        with open(file1, "r") as f1, open(file2, "r") as f2:
            tokens1 = list(filter(None, re.split(r"[ \n]", f1.read())))
            tokens2 = list(filter(None, re.split(r"[ \n]", f2.read())))

        if len(tokens1) != len(tokens2):
            print(f"{Colors.YELLOW}File token lengths differ.{Colors.RESET}")
            return False

        for t1, t2 in zip(tokens1, tokens2):
            if t1 != t2:
                try:
                    float1 = float.fromhex(t1)
                    float2 = float.fromhex(t2)
                    if abs(float1 - float2) > 1e-2:
                        print(f"{Colors.YELLOW}Floating-point difference exceeds tolerance: {t1} vs {t2}{Colors.RESET}")
                        return False
                except ValueError:
                    print(f"{Colors.YELLOW}Non-float difference: {t1} vs {t2}{Colors.RESET}")
                    return False

        return True

    except Exception as e:
        print(f"{Colors.RED}Error while comparing files {file1} and {file2}{Colors.RESET}")
        print(f"{Colors.RED}{str(e)}{Colors.RESET}")
        return False

def compile_project(timeout: int) -> None:
    """
    Compile the project using Cargo.
    """
    command = "cargo build --release"
    print(f"{Colors.BOLD}Starting compilation...{Colors.RESET}")
    result = execute_command(command, timeout)

    if result["returncode"] != 0:
        print(f"{Colors.RED}Compilation failed.{Colors.RESET}")
        print(f"{Colors.RED}{result['stderr']}{Colors.RESET}")
        exit(1)
    else:
        print(f"{Colors.GREEN}Compilation succeeded.{Colors.RESET}")
        if result["stdout"]:
            print(f"{Colors.BLUE}{result['stdout']}{Colors.RESET}")

def find_testcases(testcase_dir: str) -> List[str]:
    """
    Recursively find all test case files ending with .sy in the given directory.
    """
    testcases = []
    for root, _, files in os.walk(testcase_dir):
        for file in sorted(files):
            if file.endswith(".sy"):
                testcase_path = os.path.join(root, file)
                testcases.append(testcase_path.rsplit(".", 1)[0])
    return testcases

def test(config: Config) -> None:
    """
    Run test cases and generate a Markdown report.
    """
    testcases = find_testcases(config.testcase_dir)
    result_md = "# Test Result\n\n"
    passed = 0
    total = len(testcases)
    result_md_table = "| Testcase | Status |\n| -------- | ------ |\n"

    for testcase in testcases:
        basename = os.path.basename(testcase)
        in_path = f"{testcase}.in" if os.path.isfile(f"{testcase}.in") else None
        std_out_path = f"{testcase}.out"

        asm_path = os.path.join(config.output_dir, f"{basename}.s")
        ir_path = os.path.join(config.output_dir, f"{basename}.ll")
        exec_path = os.path.join(config.output_dir, f"{basename}")
        out_path = os.path.join(config.output_dir, f"{basename}.out")
        log_path = os.path.join(config.output_dir, f"{basename}.log")

        with open(log_path, "w") as log_file:
            if config.test_llvm:
                # Compile to LLVM IR
                # XXX: Add more arguments if you need.
                compile_command = (
                    f"{config.executable_path} -S -o {asm_path} {testcase}.sy "
                    f"--emit-llvm-ir {ir_path} -O{config.opt_level}"
                )

                log_file.write(f"Executing: {compile_command}\n")
                compile_result = execute_command(compile_command, config.timeout)
                log_file.write(f"STDOUT:\n{compile_result['stdout']}\n")
                log_file.write(f"STDERR:\n{compile_result['stderr']}\n")

                if compile_result["returncode"] != 0:
                    if compile_result["stderr"] == "TIMEOUT":
                        status = f"⚠️ orzcc TLE"
                    else:
                        status = f"⚠️ orzcc RE"
                    result_md_table += f"| `{basename}` | {status} |\n"
                    print(f"{Colors.RED}[ C E ] Compilation error for {basename}, see {log_path}{Colors.RESET}")
                    continue

                # Use LLVM to compile to assembly
                compile_command = (
                    f"clang -mllvm -opaque-pointers -fno-addrsig -S --target=riscv64-linux-gnu-gcc -mabi=lp64d {ir_path} -o {asm_path}"
                )

                log_file.write(f"Executing: {compile_command}\n")
                compile_result = execute_command(compile_command, config.timeout)
                log_file.write(f"STDOUT:\n{compile_result['stdout']}\n")
                log_file.write(f"STDERR:\n{compile_result['stderr']}\n")

                if compile_result["returncode"] != 0:
                    if compile_result["stderr"] == "TIMEOUT":
                        status = f"⚠️ llc TLE"
                    else:
                        status = f"⚠️ llc RE"
                    result_md_table += f"| `{basename}` | {status} |\n"
                    print(f"{Colors.RED}[ C E ] LLVMIR Compilation error for {basename}, see {log_path}{Colors.RESET}")
                    continue
            
            else:
                # Compile to assembly
                compile_command = (
                    f"{config.executable_path} -S -o {asm_path} {testcase}.sy "
                    f"--emit-ir {ir_path} --emit-vcode {asm_path}.vcode -O{config.opt_level}"
                )
                log_file.write(f"Executing: {compile_command}\n")
                compile_result = execute_command(compile_command, config.timeout)
                log_file.write(f"STDOUT:\n{compile_result['stdout']}\n")
                log_file.write(f"STDERR:\n{compile_result['stderr']}\n")

                if compile_result["returncode"] != 0:
                    if compile_result["stderr"] == "TIMEOUT":
                        status = f"⚠️ orzcc TLE"
                    else:
                        status = f"⚠️ orzcc RE"
                    result_md_table += f"| `{basename}` | {status} |\n"
                    print(f"{Colors.RED} [ C E ] Compilation error for {basename}, see {log_path}{Colors.RESET}")
                    continue

            # Compile assembly to executable
            gcc_command = (
                f"riscv64-linux-gnu-gcc -march=rv64gc {asm_path} "
                f"-L{config.runtime_lib_dir} -lsylib -o {exec_path}"
            )
            log_file.write(f"Executing: {gcc_command}\n")
            gcc_result = execute_command(gcc_command, config.timeout)
            log_file.write(f"STDOUT:\n{gcc_result['stdout']}\n")
            log_file.write(f"STDERR:\n{gcc_result['stderr']}\n")

            if gcc_result["returncode"] != 0:
                status = f"😢 CE"
                result_md_table += f"| `{basename}` | {status} |\n"
                print(f"{Colors.RED} [ C E ] GCC compilation error for {basename}, see {log_path}{Colors.RESET}")
                continue

            # Execute the program using QEMU
            qemu_command = (
                f"qemu-riscv64 -cpu rv64,zba=true,zbb=true -L /usr/riscv64-linux-gnu "
                f"{exec_path} "
                + (f"< {in_path} " if in_path else "") +
                f"> {out_path}"
            )
            log_file.write(f"Executing: {qemu_command}\n")
            qemu_result = execute_command(qemu_command, config.timeout)
            log_file.write(f"STDOUT:\n{qemu_result['stdout']}\n")
            log_file.write(f"STDERR:\n{qemu_result['stderr']}\n")

            need_newline = False
            with open(out_path, "r") as f:
                content = f.read()
                if len(content) > 0:
                    if not content.endswith("\n"):
                        need_newline = True

            # add return code to the last line of out file
            with open(out_path, "a+") as f:
                if need_newline:
                    f.write("\n")
                f.write(str(qemu_result["returncode"]))
                f.write("\n")

            # Check the result
            is_correct = check_file(out_path, std_out_path, f"{config.output_dir}/{basename}.diff")

            if qemu_result["returncode"] is None:
                if qemu_result["stderr"] == "TIMEOUT":
                    status = f"⏱️ TLE"
                    print(f"{Colors.YELLOW}[ TLE ] Test case {basename} timed out.{Colors.RESET}")
                else:
                    status = f"🆘 RE"
                    print(f"{Colors.RED}[ R E ] Runtime error in test case {basename}.{Colors.RESET}")
            elif is_correct:
                status = f"✅ AC"
                passed += 1
                print(f"{Colors.GREEN}[ A C ] Test case {basename} passed.{Colors.RESET}")
            else:
                status = f"❌ WA"
                print(f"{Colors.RED}[ W A ] Test case {basename} failed. See {log_path} and {config.output_dir}/{basename}.diff .{Colors.RESET}")

            result_md_table += f"| `{basename}` | {status} |\n"

    # Summary
    result_md += f"Passed {passed}/{total} testcases.\n\n"
    print(f"{Colors.BOLD}Passed {passed}/{total} testcases.{Colors.RESET}")
    result_md += result_md_table

    # Write Markdown report
    report_path = os.path.join(config.output_dir, "result.md")
    try:
        with open(report_path, "w") as f:
            f.write(result_md)
        print(f"{Colors.GREEN}Test report generated at {report_path}{Colors.RESET}")
    except Exception as e:
        print(f"{Colors.RED}Failed to write test report to {report_path}{Colors.RESET}")
        print(f"{Colors.RED}{str(e)}{Colors.RESET}")

def main() -> None:
    config = parse_args()
    os.makedirs(config.output_dir, exist_ok=True)

    if not config.no_compile:
        compile_project(config.timeout)

    if not config.no_test:
        if os.path.exists(config.output_dir):
            shutil.rmtree(config.output_dir)
        os.makedirs(config.output_dir, exist_ok=True)
        
        test(config)

if __name__ == "__main__":
    main()
