# s05_skill_loading.py - 技能加载（Skills）

## 核心概念

技能加载机制实现**按需知识注入**，通过两层架构避免系统提示词膨胀：
- **第一层（廉价）**：技能名称和描述，放入系统提示词
- **第二层（按需）**：完整技能内容，通过工具结果返回

## 架构图

```
skills/
  pdf/
    SKILL.md          <-- frontmatter (name, description) + body
  code-review/
    SKILL.md

系统提示词:
+--------------------------------------+
| You are a coding agent.              |
| Skills available:                    |
|   - pdf: Process PDF files...        |  <-- Layer 1: 元数据
|   - code-review: Review code...     |
+--------------------------------------+

当模型调用 load_skill("pdf"):
+--------------------------------------+
| tool_result:                         |
| <skill name="pdf">                   |
|   Full PDF processing instructions   |  <-- Layer 2: 完整内容
|   Step 1: ...                        |
|   Step 2: ...                        |
| </skill>                             |
+--------------------------------------+
```

## 关键特性

### 1. 双层技能加载

#### Layer 1: 元数据层
- 在 `SkillLoader.get_descriptions()` 中生成
- 放入系统提示词，供模型了解可用技能
- 每个技能约100 tokens，开销极低

#### Layer 2: 内容层
- 在 `SkillLoader.get_content()` 中按需加载
- 通过 `load_skill` 工具返回完整技能内容
- 使用 `<skill>` XML 标签包裹，便于模型识别

### 2. 技能文件格式

使用 YAML frontmatter 格式：

```yaml
---
name: pdf
description: Process PDF files with OCR
tags: document, ocr, extraction
---
# 技能正文内容
Step 1: Install required libraries...
Step 2: Use PyPDF2 to extract text...
Step 3: Apply OCR if needed...
```

### 3. SkillLoader 类

```python
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills = {}
        self._load_all()  # 扫描所有 SKILL.md

    def get_descriptions(self) -> str:
        """Layer 1: 生成技能描述列表"""
        
    def get_content(self, name: str) -> str:
        """Layer 2: 获取完整技能内容"""
```

## 工具定义

| 工具名 | 功能 |
|--------|------|
| `bash` | 执行shell命令 |
| `read_file` | 读取文件内容 |
| `write_file` | 写入文件内容 |
| `edit_file` | 编辑文件内容 |
| `load_skill` | **按需加载技能知识** |

## 安全机制

- 路径安全检查：防止目录遍历攻击
- 危险命令拦截：阻止高危shell命令
- 执行超时限制：120秒

## 使用方式

### 1. 创建技能目录
```bash
mkdir -p skills/pdf
mkdir -p skills/code-review
```

### 2. 编写技能文件（skills/pdf/SKILL.md）
```yaml
---
name: pdf
description: Process PDF files
---
Step 1: Check if PDF is text-based or scanned...
Step 2: Use PyPDF2 for text extraction...
Step 3: Use pytesseract for OCR on scanned PDFs...
```

### 3. 运行
```bash
python s05_skill_loading.py
```

## 核心洞察

> **"不要把所有知识都放入系统提示词，按需加载。"**

按需加载策略的优势：
1. **减少上下文开销**：系统提示词保持精简
2. **动态加载**：只在需要时获取技能
3. **可扩展性**：新增技能只需添加文件，无需修改代码

## 文件结构

| 组件 | 描述 |
|------|------|
| `SkillLoader` | 技能加载器类 |
| `SKILLS_DIR` | 技能目录路径 |
| `SYSTEM` | 包含Layer 1信息的系统提示词 |
| `TOOL_HANDLERS` | 工具处理器映射 |
| `TOOLS` | 工具定义列表 |