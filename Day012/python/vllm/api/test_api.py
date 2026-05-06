"""
vLLM Serve API 测试脚本 - 使用确定性参数
"""
import subprocess
import time
import requests
import json
import signal
import sys

MODEL_PATH = "/data/Workspace/airelearn/Day012/python/vllm/cache/storier"
API_PORT = 8000


class VLLMServer:
    """vLLM 服务器管理器"""

    def __init__(self, model_path: str, port: int = 8000):
        self.model_path = model_path
        self.port = port
        self.process = None

    def start(self, timeout: int = 120) -> bool:
        """启动 vLLM serve 服务器"""
        cmd = [
            "vllm", "serve",
            self.model_path,
            "--dtype", "float32",
            "--max-model-len", "512",
            "--trust-remote-code",
            "--port", str(self.port),
        ]
        print(f"启动命令: {' '.join(cmd)}")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                resp = requests.get(f"http://localhost:{self.port}/v1/models", timeout=2)
                if resp.status_code == 200:
                    print(f"服务器已就绪 (耗时 {time.time() - start_time:.1f}s)")
                    return True
            except requests.exceptions.RequestException:
                pass

            if self.process.poll() is not None:
                print("服务器进程已退出!")
                self._print_output()
                return False

            time.sleep(2)

        print("服务器启动超时!")
        self._print_output()
        return False

    def _print_output(self):
        if self.process and self.process.stdout:
            remaining = self.process.stdout.read()
            if remaining:
                print("\n=== 服务器输出 ===")
                print(remaining)

    def stop(self):
        if self.process:
            print("\n正在停止服务器...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            print("服务器已停止")


def test_completion_api(base_url: str) -> bool:
    """测试文本补全 API - 使用确定性参数"""
    print("\n=== 测试 /v1/completions (确定性参数) ===")
    # 使用与 LLM 类相同的参数
    payload = {
        "model": MODEL_PATH,
        "prompt": "凌风渊",
        "max_tokens": 20,
        "temperature": 0.8,
        "top_k": 5,
        "seed": 0,  # 设置固定 seed
    }
    try:
        resp = requests.post(
            f"{base_url}/v1/completions",
            json=payload,
            timeout=60
        )
        print(f"状态码: {resp.status_code}")
        if resp.status_code == 200:
            result = resp.json()
            print(f"补全结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            # 验证结果
            text = result["choices"][0]["text"]
            expected = " 姐姐 姐姐主人 是是 好 的 哥 阿女我 "
            if text == expected:
                print(f"\n✓ 结果匹配! 预期: {repr(expected)}")
                return True
            else:
                print(f"\n✗ 结果不匹配!")
                print(f"  预期: {repr(expected)}")
                print(f"  实际: {repr(text)}")
                return False
        else:
            print(f"错误: {resp.text}")
            return False
    except Exception as e:
        print(f"请求失败: {e}")
        return False


def test_chat_api(base_url: str) -> bool:
    """测试聊天 API - 使用确定性参数"""
    print("\n=== 测试 /v1/chat/completions (确定性参数) ===")
    # 聊天 API 不支持 seed，所以结果可能不同
    payload = {
        "model": MODEL_PATH,
        "messages": [
            {"role": "user", "content": "凌风渊"}
        ],
        "max_tokens": 20,
        "temperature": 0.8,
    }
    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        print(f"状态码: {resp.status_code}")
        if resp.status_code == 200:
            result = resp.json()
            print(f"聊天结果: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"错误: {resp.text}")
            return False
    except Exception as e:
        print(f"请求失败: {e}")
        return False


def main():
    print("=" * 60)
    print("vLLM Serve API 测试 (确定性参数)")
    print("=" * 60)

    server = VLLMServer(MODEL_PATH, API_PORT)
    base_url = f"http://localhost:{API_PORT}"

    try:
        if not server.start(timeout=120):
            print("\n服务器启动失败!")
            sys.exit(1)

        print("\n服务器启动成功!")

        # 运行测试
        results = {
            "completion_api": test_completion_api(base_url),
            "chat_api": test_chat_api(base_url),
        }

        print("\n" + "=" * 60)
        print("测试结果摘要")
        print("=" * 60)
        for test_name, passed in results.items():
            status = "PASS" if passed else "FAIL"
            print(f"{test_name}: {status}")

        all_passed = all(results.values())
        print("\n" + ("全部测试通过!" if all_passed else "部分测试失败!"))

    except KeyboardInterrupt:
        print("\n\n测试被中断")
    finally:
        server.stop()

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
