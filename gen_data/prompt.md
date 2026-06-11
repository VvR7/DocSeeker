## 目标
生成一个文档QA的多选题benchmark

### 源数据
`./paper_img/{paper_name}`下是一个paper所有页的图片image，对于第`page_idx`页，文件名为f"page{page_idx}.jpg"

### 数据生成pipeline
对于每个paper:`{paper_name}`:
- 遍历每一页，提取出`page{page_idx}.jpg`，作为图片信息给模型

先构造prompt让模型判断这一页是不是"参考文献页"，因为参考文献在论文里是没意义的，不对只包含参考文献信息的页内容设置问题。

若不是参考文献页，则：
- 使用`prompt_mm.py`的prompt，让模型生成图片/表格/公式相关的题目
  - prompt里面还涉及，若当前这页是纯文本， 会输出`{"result": "NO"}`，你要注意
- 使用`prompt_text.py`的prompt，让模型生成文本内容相关的题目
- 生成的题目实时记录在json文件中
- 对于每道生成的题目，构造prompt给模型做review：
  - 1.将当前这张图片+问题作为上下文，给模型答一遍
  - 2.只将问题作为上下文，给模型答一遍(即不可见这个图片)
  - 若1中模型答对，2中模型答错，则将此问题加入到一个新json中:`fliter.json`
  - 否则也要保留原始问题(不要删掉)，并对每个问题也在json中标记1和2的结果

json中的每个条目至少要记录：
- 出处，即`{paper_name}`
- 问题类型(text、公式、图片、表格)
- 问题
- 选项
- ground_truth
- 页码(`page_idx`)，便于定位答案

### 代码编写要求
- 模型使用Qwen3VL，推理选择vllm方式：一个参考脚本在`/HOME/sysu_gbli2/sysu_gbli2xy_1/HDD_POOL/zdw/Doc/gen_data/vllm/vllm_local.py`，你需要修改一下
- 要有详细的log日志，方便我追踪benchmark数据生成情况
- 对于模型的输出，要检查是否符合格式