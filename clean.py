import os

# 获取当前目录下的所有文件和目录
files = os.listdir('.')

# 遍历所有条目
for file in files:
    #
    if file.startswith('gpu') and os.path.isfile(file):
        try:
            # 删除文件
            os.remove(file)
            print(f"已删除文件: {file}")
        except Exception as e:
            print(f"删除文件 {file} 失败: {e}")

print("操作完成")

# 遍历所有条目
for file in files:
    # 检查是否是以'my_log'开头的文件（排除目录）
    if file.startswith('_') and os.path.isdir(file):
        try:
            # 删除文件
            os.removedirs(file)
            print(f"已删除文件: {file}")
        except Exception as e:
            print(f"删除文件 {file} 失败: {e}")

print("操作完成")

# 遍历所有条目
for file in files:
    # 检查是开头的文件（排除目录）
    if file.startswith('gpu') and os.path.isfile(file):
        try:
            # 删除文件
            os.remove(file)
            print(f"已删除文件: {file}")
        except Exception as e:
            print(f"删除文件 {file} 失败: {e}")

print("操作完成")
