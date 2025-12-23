# 1. 导入必要的库
import oss2  # 阿里云OSS SDK
import os    # 用于处理文件路径
from pathlib import Path  # 更安全的路径处理

# -------------------------- 2. 配置信息（必须替换成你的！） --------------------------
# 阿里云OSS凭证
ACCESS_KEY_ID = "LTAI5t86RbFrzzyTUnqussE3"       # 替换：步骤2.4获取的AccessKey ID
ACCESS_KEY_SECRET = "HAunjuGb98whAmvzx2AX34pvmPeBxO" # 替换：步骤2.4获取的AccessKey Secret
ENDPOINT = "oss-cn-beijing.aliyuncs.com" # 替换：步骤2.3获取的Endpoint
BUCKET_NAME = "zshidai"          # 替换：步骤2.2创建的Bucket名称

# 路径配置
LOCAL_MODEL_DIR = "./model"  # 本地模型目录（你的项目中model文件夹的路径，无需修改）
OSS_MODEL_DIR = "model/"     # OSS上的目标目录（保持和本地一致，方便后续查找）
# -----------------------------------------------------------------------------------

# 3. 初始化OSS客户端
def init_oss_client():
    # 验证配置是否完整
    if not all([ACCESS_KEY_ID, ACCESS_KEY_SECRET, ENDPOINT, BUCKET_NAME]):
        raise ValueError("请先填写完整的OSS配置信息！")
    # 创建OSS认证对象
    auth = oss2.Auth(ACCESS_KEY_ID, ACCESS_KEY_SECRET)
    # 连接到指定Bucket
    bucket = oss2.Bucket(auth, ENDPOINT, BUCKET_NAME)
    return bucket

# 4. 批量上传模型文件
def upload_models():
    # 初始化OSS客户端
    bucket = init_oss_client()
    
    # 检查本地模型目录是否存在
    local_model_path = Path(LOCAL_MODEL_DIR)
    if not local_model_path.exists() or not local_model_path.is_dir():
        print(f"❌ 本地模型目录不存在：{LOCAL_MODEL_DIR}")
        return
    
    # 遍历本地model目录下的所有.pkl文件
    for file_path in local_model_path.glob("*.pkl"):
        # 获取文件名（如"zgen_preference_model.pkl"）
        file_name = file_path.name
        # 拼接OSS上的目标路径（如"model/zgen_preference_model.pkl"）
        oss_target_path = os.path.join(OSS_MODEL_DIR, file_name)
        
        try:
            # 上传本地文件到OSS
            bucket.put_object_from_file(oss_target_path, str(file_path))
            print(f"✅ 上传成功：{file_path} → OSS路径：{oss_target_path}")
        except Exception as e:
            print(f"❌ 上传失败：{file_path}，错误：{str(e)}")

# 5. 执行上传
if __name__ == "__main__":
    print("开始批量上传模型文件到阿里云OSS...")
    upload_models()
    print("上传任务结束！")
