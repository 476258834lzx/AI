"""
vLLM serve 测试脚本

使用方法：
1. 确保 model.py 已复制到模型目录
2. 运行本脚本启动 vLLM serve
"""
import subprocess
import time
import requests
import json

MODEL_PATH = "/data/Workspace/airelearn/Day012/python/vllm/cache/storier"
API_PORT = 8000

def start_server():
    """启动 vLLM serve 服务器"""
    cmd = [
        "vllm", "serve",
        MODEL_PATH,
        "--dtype", "float32",
        "--max-model-len", "512",
        "--trust-remote-code",
        "--port", str(API_PORT),
    ]
    print(f"启动命令: {' '.join(cmd)}")
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

def wait_for_server(timeout=120):
    """等待服务器启动"""
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = requests.get(f"http://localhost:{API_PORT}/v1/models", timeout=2)
            if resp.status_code == 200:
                print(f"服务器已就绪 (耗时 {time.time() - start:.1f}s)")
                return True
        except:
            pass
        time.sleep(2)
    return False

def test_completion():
    """测试文本补全 API"""
    print("\n测试补全 API...")
    payload = {
        "model": MODEL_PATH,
        "prompt": "你好，",
        "max_tokens": 20,
        "temperature": 0.8,
    }
    try:
        resp = requests.post(
            f"http://localhost:{API_PORT}/v1/completions",
            json=payload,
            timeout=60
        )
        print(f"状态码: {resp.status_code}")
        if resp.status_code == 200:
            result = resp.json()
            print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
            return True
        else:
            print(f"错误: {resp.text}")
            return False
    except Exception as e:
        print(f"请求失败: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("vLLM Serve 测试")
    print("=" * 60)

    proc = start_server()
    try:
        if wait_for_server():
            print("服务器启动成功！")
            test_completion()
        else:
            print("服务器启动超时！")
    finally:
        print("\n正在停止服务器...")
        proc.terminate()
        proc.wait()
        print("服务器已停止")
