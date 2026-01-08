"""
MinerU VLM 生产环境 API 封装
支持人工干预的文档版面识别和内容提取工作流

工作流程：
1. 调用 detect_layout() 识别版面，返回可编辑的区块列表
2. 前端展示区块，用户调整：区块范围、类型标签、旋转角度
3. 调用 extract_blocks_content() 批量提取用户确认/修改后的区块内容
4. 用户修正提取的文本内容

作者：AI Assistant
日期：2026-01-08
"""

import json
from typing import Literal, Optional
from dataclasses import dataclass, asdict
from PIL import Image

from mineru_vl_utils.mineru_client import MinerUClient
from mineru_vl_utils.structs import ContentBlock


# ==================== 数据结构定义 ====================

@dataclass
class LayoutBlock:
    """
    版面布局区块数据结构（用于前后端交互）
    
    属性说明：
        block_id: 区块唯一标识符（便于前端定位和修改）
        type: 区块类型，可选值：
              - text: 普通文本
              - title: 标题
              - table: 表格
              - equation: 公式
              - figure: 图片
              - caption: 图表标题
              - footer: 页脚
              - header: 页眉
              - equation_block: 公式块
        bbox: 边界框坐标 [x1, y1, x2, y2]，归一化到 0-1 范围
              x1,y1: 左上角坐标
              x2,y2: 右下角坐标
        angle: 旋转角度，可选值：0, 90, 180, 270（顺时针度数）
        confidence: 置信度（预留字段，当前版本可能为None）
        content: 区块内容（初始为None，提取后填充）
    """
    block_id: str
    type: str
    bbox: list[float]  # [x1, y1, x2, y2] 归一化坐标
    angle: int  # 0, 90, 180, 270
    confidence: Optional[float] = None
    content: Optional[str] = None
    
    def to_dict(self):
        """转换为字典格式，便于JSON序列化"""
        return asdict(self)
    
    def to_content_block(self) -> ContentBlock:
        """转换为MinerU内部的ContentBlock格式"""
        return ContentBlock(
            type=self.type,
            bbox=self.bbox,
            angle=self.angle,
            content=self.content
        )
    
    @classmethod
    def from_content_block(cls, block: ContentBlock, block_id: str):
        """从MinerU的ContentBlock转换"""
        return cls(
            block_id=block_id,
            type=block.type,
            bbox=block.bbox,
            angle=block.angle if block.angle else 0,
            confidence=None,
            content=block.content
        )


@dataclass
class LayoutDetectionResult:
    """
    版面识别结果
    
    属性说明：
        image_id: 图片唯一标识符（关联原始图片）
        image_width: 原始图片宽度（像素）
        image_height: 原始图片高度（像素）
        blocks: 检测到的区块列表
        total_blocks: 区块总数
    """
    image_id: str
    image_width: int
    image_height: int
    blocks: list[LayoutBlock]
    total_blocks: int
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "image_id": self.image_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "blocks": [block.to_dict() for block in self.blocks],
            "total_blocks": self.total_blocks
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


@dataclass
class BlockExtractionRequest:
    """
    区块内容提取请求（用户修改后的区块信息）
    
    属性说明：
        image_id: 图片标识符（需要与原始图片对应）
        blocks: 用户确认/修改后的区块列表
    """
    image_id: str
    blocks: list[LayoutBlock]
    
    @classmethod
    def from_dict(cls, data: dict):
        """从字典创建请求对象"""
        blocks = [
            LayoutBlock(**block_data) 
            for block_data in data["blocks"]
        ]
        return cls(
            image_id=data["image_id"],
            blocks=blocks
        )


@dataclass
class BlockExtractionResult:
    """
    区块内容提取结果
    
    属性说明：
        image_id: 图片标识符
        blocks: 包含提取内容的区块列表
        success_count: 成功提取的区块数量
        failed_count: 失败的区块数量
    """
    image_id: str
    blocks: list[LayoutBlock]
    success_count: int
    failed_count: int
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            "image_id": self.image_id,
            "blocks": [block.to_dict() for block in self.blocks],
            "success_count": self.success_count,
            "failed_count": self.failed_count
        }
    
    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


# ==================== 核心 API 类 ====================

class MinerUProductionAPI:
    """
    MinerU 生产环境 API 封装
    
    提供两个核心接口：
    1. detect_layout() - 版面识别
    2. extract_blocks_content() - 批量区块内容提取
    
    使用示例：
        # 初始化
        api = MinerUProductionAPI(
            backend="http-client",
            server_url="http://localhost:8000"
        )
        
        # 步骤1：识别版面
        result = api.detect_layout(image, image_id="page_001")
        
        # 步骤2：前端展示，用户调整区块
        # ... 用户在UI上修改区块范围、类型、角度 ...
        
        # 步骤3：提取用户确认后的区块内容
        modified_blocks = [...]  # 用户修改后的区块
        extraction_result = api.extract_blocks_content(
            image=image,
            blocks=modified_blocks,
            image_id="page_001"
        )
    """
    
    def __init__(
        self,
        backend: Literal[
            "http-client",
            "transformers",
            "mlx-engine",
            "lmdeploy-engine",
            "vllm-engine",
            "vllm-async-engine",
        ],
        model_name: Optional[str] = None,
        server_url: Optional[str] = None,
        server_headers: Optional[dict[str, str]] = None,
        model_path: Optional[str] = None,
        # 以下为高级配置参数
        layout_image_size: tuple[int, int] = (1036, 1036),
        min_image_edge: int = 28,
        max_image_edge_ratio: float = 50,
        simple_post_process: bool = False,
        handle_equation_block: bool = True,
        abandon_list: bool = False,
        abandon_paratext: bool = False,
        max_concurrency: int = 100,
        http_timeout: int = 600,
        max_retries: int = 3,
        retry_backoff_factor: float = 0.5,
        debug: bool = False,
    ):
        """
        初始化 MinerU 客户端
        
        参数说明：
            backend: 推理后端类型
                - "http-client": HTTP API 调用（推荐生产环境）
                - "transformers": 本地 Transformers 模型
                - "vllm-engine": vLLM 推理引擎
                - 其他: mlx/lmdeploy 等
                
            server_url: API 服务地址（backend="http-client"时必填）
                例如：http://localhost:8000
                
            server_headers: HTTP 请求头（可选）
                例如：{"Authorization": "Bearer token"}
                
            model_path: 模型路径（本地推理时必填）
            
            layout_image_size: 布局检测时图片缩放尺寸
                默认 (1036, 1036)，影响检测精度和速度
                
            min_image_edge: 最小图片边长（像素）
                小于此值会被放大，避免过小图片识别失败
                
            max_image_edge_ratio: 最大宽高比
                超过会进行padding，避免图片变形
                
            simple_post_process: 是否使用简化后处理
                False: 完整后处理（合并段落、处理列表等）
                True: 仅基础处理
                
            handle_equation_block: 是否处理公式块
            abandon_list: 是否丢弃列表内容
            abandon_paratext: 是否丢弃段落文本
            
            max_concurrency: 最大并发数（批量处理时）
            http_timeout: HTTP 请求超时时间（秒）
            max_retries: 最大重试次数
            retry_backoff_factor: 重试退避因子
            debug: 是否开启调试模式
        """
        self.client = MinerUClient(
            backend=backend,
            model_name=model_name,
            server_url=server_url,
            server_headers=server_headers,
            model_path=model_path,
            layout_image_size=layout_image_size,
            min_image_edge=min_image_edge,
            max_image_edge_ratio=max_image_edge_ratio,
            simple_post_process=simple_post_process,
            handle_equation_block=handle_equation_block,
            abandon_list=abandon_list,
            abandon_paratext=abandon_paratext,
            max_concurrency=max_concurrency,
            http_timeout=http_timeout,
            max_retries=max_retries,
            retry_backoff_factor=retry_backoff_factor,
            debug=debug,
        )
        self.debug = debug
    
    def detect_layout(
        self, 
        image: Image.Image,
        image_id: str,
        priority: Optional[int] = None
    ) -> LayoutDetectionResult:
        """
        接口1：版面识别
        
        功能说明：
            - 检测文档图片中的所有区块（文本、表格、公式等）
            - 返回每个区块的位置、类型、旋转角度
            - 不提取具体内容（内容为None），节省计算资源
        
        参数：
            image: PIL.Image 对象
            image_id: 图片唯一标识符（用于后续关联）
            priority: 优先级（可选，用于任务调度）
        
        返回：
            LayoutDetectionResult 对象，包含：
                - image_id: 图片标识
                - image_width, image_height: 图片尺寸
                - blocks: 区块列表（每个区块包含位置、类型、角度）
                - total_blocks: 区块总数
        
        使用示例：
            image = Image.open("document.png")
            result = api.detect_layout(image, image_id="doc_page1")
            
            # 转换为JSON传给前端
            json_data = result.to_json()
            
            # 或直接操作区块列表
            for block in result.blocks:
                print(f"区块 {block.block_id}: {block.type} at {block.bbox}")
        
        注意事项：
            1. 返回的区块content字段为None，需要调用extract_blocks_content()填充
            2. bbox坐标已归一化到0-1范围，前端显示时需乘以图片实际尺寸
            3. angle为顺时针旋转角度（0/90/180/270）
        """
        # 获取图片尺寸
        width, height = image.size
        
        # 调用 MinerU 布局检测（仅检测位置，不提取内容）
        if self.debug:
            print(f"[DEBUG] 开始检测版面：image_id={image_id}, size={width}x{height}")
        
        content_blocks = self.client.layout_detect(image, priority=priority)
        
        if self.debug:
            print(f"[DEBUG] 检测到 {len(content_blocks)} 个区块")
        
        # 转换为生产API格式
        layout_blocks = []
        for idx, block in enumerate(content_blocks):
            block_id = f"{image_id}_block_{idx:03d}"
            layout_block = LayoutBlock.from_content_block(block, block_id)
            layout_blocks.append(layout_block)
            
            if self.debug:
                print(f"[DEBUG] 区块 {block_id}: type={block.type}, "
                      f"bbox={block.bbox}, angle={block.angle}")
        
        result = LayoutDetectionResult(
            image_id=image_id,
            image_width=width,
            image_height=height,
            blocks=layout_blocks,
            total_blocks=len(layout_blocks)
        )
        
        return result
    
    def extract_blocks_content(
        self,
        image: Image.Image,
        blocks: list[LayoutBlock],
        image_id: str,
        priority: Optional[int] = None,
        skip_types: Optional[list[str]] = None
    ) -> BlockExtractionResult:
        """
        接口2：批量提取区块内容
        
        功能说明：
            - 根据用户确认/修改后的区块信息，批量提取文本内容
            - 支持用户调整后的区块范围、类型、旋转角度
            - 自动跳过不需要提取内容的区块类型（如image）
        
        参数：
            image: PIL.Image 对象（必须与detect_layout时的图片对应）
            blocks: 用户确认/修改后的区块列表
            image_id: 图片标识符（用于日志和调试）
            priority: 优先级（可选）
            skip_types: 不需要提取内容的区块类型列表（可选）
                默认跳过：["image", "figure"]
        
        返回：
            BlockExtractionResult 对象，包含：
                - image_id: 图片标识
                - blocks: 填充了content字段的区块列表
                - success_count: 成功提取的区块数
                - failed_count: 失败的区块数
        
        使用示例：
            # 场景1：用户修改了某些区块后重新提取
            modified_blocks = [
                LayoutBlock(
                    block_id="page1_block_001",
                    type="table",  # 用户将类型改为table
                    bbox=[0.1, 0.2, 0.9, 0.5],  # 用户调整了范围
                    angle=90  # 用户旋转了角度
                ),
                # ... 更多区块
            ]
            
            result = api.extract_blocks_content(
                image=image,
                blocks=modified_blocks,
                image_id="page1"
            )
            
            # 场景2：仅提取特定区块
            table_blocks = [b for b in all_blocks if b.type == "table"]
            result = api.extract_blocks_content(
                image=image,
                blocks=table_blocks,
                image_id="page1"
            )
            
            # 获取提取结果
            for block in result.blocks:
                if block.content:
                    print(f"{block.block_id}: {block.content}")
        
        注意事项：
            1. blocks参数可以是detect_layout返回的全部区块，也可以是用户筛选后的部分区块
            2. 系统自动跳过image/figure类型区块（这些区块不包含可提取文本）
            3. 如果某个区块提取失败，其content字段保持为None，但不影响其他区块
            4. bbox坐标应为归一化坐标（0-1范围），系统会自动转换为像素坐标
            5. 支持用户修改的type/bbox/angle，会按新参数进行提取
        """
        # 默认跳过的类型
        if skip_types is None:
            skip_types = ["image", "figure"]
        
        if self.debug:
            print(f"[DEBUG] 开始批量提取区块内容：image_id={image_id}, "
                  f"total_blocks={len(blocks)}")
        
        # 转换为 MinerU 内部格式
        content_blocks = [block.to_content_block() for block in blocks]
        
        # 调用 MinerU 的准备函数（裁剪图片、准备提示词等）
        block_images, prompts, sampling_params, indices = \
            self.client.helper.prepare_for_extract(
                image=image,
                blocks=content_blocks,
                not_extract_list=skip_types
            )
        
        if self.debug:
            print(f"[DEBUG] 准备提取 {len(block_images)} 个区块"
                  f"（跳过了 {len(blocks) - len(block_images)} 个）")
        
        # 批量调用模型提取内容
        if not block_images:
            # 所有区块都被跳过
            if self.debug:
                print(f"[DEBUG] 没有需要提取的区块")
            return BlockExtractionResult(
                image_id=image_id,
                blocks=blocks,
                success_count=0,
                failed_count=0
            )
        
        outputs = self.client.client.batch_predict(
            block_images, 
            prompts, 
            sampling_params, 
            priority
        )
        
        # 将提取结果填充到对应区块
        success_count = 0
        failed_count = 0
        
        for idx, output in zip(indices, outputs):
            content_blocks[idx].content = output
            blocks[idx].content = output
            
            if output and output.strip():
                success_count += 1
                if self.debug:
                    preview = output[:50] + "..." if len(output) > 50 else output
                    print(f"[DEBUG] 区块 {blocks[idx].block_id} 提取成功: {preview}")
            else:
                failed_count += 1
                if self.debug:
                    print(f"[DEBUG] 区块 {blocks[idx].block_id} 提取失败或内容为空")
        
        # 后处理（合并段落、处理公式等）
        if self.debug:
            print(f"[DEBUG] 开始后处理")
        
        processed_blocks = self.client.helper.post_process(content_blocks)
        
        # 更新blocks中的内容（后处理可能修改了content）
        for idx, processed_block in enumerate(processed_blocks):
            if idx < len(blocks):
                blocks[idx].content = processed_block.content
        
        result = BlockExtractionResult(
            image_id=image_id,
            blocks=blocks,
            success_count=success_count,
            failed_count=failed_count
        )
        
        if self.debug:
            print(f"[DEBUG] 提取完成：成功={success_count}, 失败={failed_count}")
        
        return result
    
    def batch_detect_layout(
        self,
        images: list[Image.Image],
        image_ids: list[str],
        priority: Optional[list[int]] = None
    ) -> list[LayoutDetectionResult]:
        """
        批量版面识别（可选接口）
        
        参数：
            images: 图片列表
            image_ids: 图片ID列表（需与images一一对应）
            priority: 优先级列表（可选）
        
        返回：
            LayoutDetectionResult 列表
        """
        if len(images) != len(image_ids):
            raise ValueError("images和image_ids长度必须一致")
        
        results = []
        for i, (image, image_id) in enumerate(zip(images, image_ids)):
            priority_val = priority[i] if priority else None
            result = self.detect_layout(image, image_id, priority_val)
            results.append(result)
        
        return results


# ==================== 工具函数 ====================

def visualize_layout(
    image: Image.Image,
    layout_result: LayoutDetectionResult,
    output_path: str
):
    """
    可视化版面检测结果（将区块框绘制在图片上）
    
    参数：
        image: 原始图片
        layout_result: 版面检测结果
        output_path: 输出图片路径
    
    注意：需要安装 pillow 和 matplotlib（可选）
    """
    from PIL import ImageDraw, ImageFont
    
    # 复制图片避免修改原图
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    
    width, height = image.size
    
    # 颜色映射（不同类型使用不同颜色）
    color_map = {
        "text": "blue",
        "title": "red",
        "table": "green",
        "equation": "purple",
        "figure": "orange",
        "image": "orange",
        "caption": "cyan",
    }
    
    for block in layout_result.blocks:
        # 转换归一化坐标为像素坐标
        x1, y1, x2, y2 = block.bbox
        x1, x2 = int(x1 * width), int(x2 * width)
        y1, y2 = int(y1 * height), int(y2 * height)
        
        # 绘制矩形框
        color = color_map.get(block.type, "gray")
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # 绘制标签
        label = f"{block.type}"
        if block.angle != 0:
            label += f" ({block.angle}°)"
        draw.text((x1, y1 - 15), label, fill=color)
    
    img_draw.save(output_path)
    print(f"可视化结果已保存到: {output_path}")


# ==================== 使用示例 ====================

if __name__ == "__main__":
    """
    完整使用示例：模拟生产环境的人工干预工作流
    """
    
    # ========== 初始化 API ==========
    api = MinerUProductionAPI(
        backend="http-client",
        server_url="http://localhost:8000",  # 替换为实际服务地址
        debug=True  # 开启调试模式查看详细日志
    )
    
    # ========== 步骤1：识别版面 ==========
    print("\n" + "="*60)
    print("步骤1：识别文档版面")
    print("="*60)
    
    # 加载图片
    image = Image.open("example_document.png")
    
    # 调用版面识别接口
    layout_result = api.detect_layout(
        image=image,
        image_id="doc_page_001"
    )
    
    print(f"\n识别结果：")
    print(f"  图片尺寸: {layout_result.image_width} x {layout_result.image_height}")
    print(f"  检测到区块数: {layout_result.total_blocks}")
    print(f"\n区块详情：")
    for block in layout_result.blocks:
        print(f"  - {block.block_id}: {block.type}, bbox={block.bbox}, angle={block.angle}")
    
    # 保存为JSON（传给前端）
    json_str = layout_result.to_json()
    with open("layout_result.json", "w", encoding="utf-8") as f:
        f.write(json_str)
    print(f"\n结果已保存到: layout_result.json")
    
    # 可视化（可选）
    # visualize_layout(image, layout_result, "layout_visualization.png")
    
    # ========== 步骤2：模拟用户修改区块 ==========
    print("\n" + "="*60)
    print("步骤2：用户在前端调整区块")
    print("="*60)
    
    # 假设用户修改了第一个区块
    modified_blocks = layout_result.blocks.copy()
    
    # 修改示例1：调整区块范围
    modified_blocks[0].bbox = [0.1, 0.15, 0.9, 0.35]  # 扩大了范围
    print(f"\n用户操作：调整了区块 {modified_blocks[0].block_id} 的范围")
    
    # 修改示例2：更改区块类型
    if len(modified_blocks) > 1:
        modified_blocks[1].type = "table"  # 原本可能识别错了，用户改为table
        print(f"用户操作：将区块 {modified_blocks[1].block_id} 的类型改为 table")
    
    # 修改示例3：旋转区块
    if len(modified_blocks) > 2:
        modified_blocks[2].angle = 90  # 用户发现图片需要旋转
        print(f"用户操作：将区块 {modified_blocks[2].block_id} 旋转 90°")
    
    # ========== 步骤3：提取用户确认后的区块内容 ==========
    print("\n" + "="*60)
    print("步骤3：提取区块内容")
    print("="*60)
    
    # 场景A：提取所有区块
    extraction_result = api.extract_blocks_content(
        image=image,
        blocks=modified_blocks,
        image_id="doc_page_001"
    )
    
    print(f"\n提取结果：")
    print(f"  成功: {extraction_result.success_count} 个")
    print(f"  失败: {extraction_result.failed_count} 个")
    print(f"\n内容详情：")
    for block in extraction_result.blocks:
        if block.content:
            preview = block.content[:80] + "..." if len(block.content) > 80 else block.content
            print(f"  {block.block_id} ({block.type}):")
            print(f"    {preview}")
    
    # 保存结果
    result_json = extraction_result.to_json()
    with open("extraction_result.json", "w", encoding="utf-8") as f:
        f.write(result_json)
    print(f"\n结果已保存到: extraction_result.json")
    
    # 场景B：仅重新提取用户修改的特定区块
    print("\n" + "="*60)
    print("步骤3.2：仅重新提取特定区块（用户修改后）")
    print("="*60)
    
    # 假设用户只对前3个区块不满意，修改后重新提取
    blocks_to_reextract = modified_blocks[:3]
    
    reextraction_result = api.extract_blocks_content(
        image=image,
        blocks=blocks_to_reextract,
        image_id="doc_page_001"
    )
    
    print(f"\n重新提取了 {len(blocks_to_reextract)} 个区块")
    
    # ========== 步骤4：用户修正提取的文本内容 ==========
    print("\n" + "="*60)
    print("步骤4：用户修正文本内容")
    print("="*60)
    
    # 模拟用户修正内容
    for block in extraction_result.blocks:
        if block.content and "错别字" in block.content:  # 假设发现了错别字
            block.content = block.content.replace("错别字", "正确字")
            print(f"用户修正: {block.block_id} 的内容")
    
    # 保存最终版本
    final_json = extraction_result.to_json()
    with open("final_result.json", "w", encoding="utf-8") as f:
        f.write(final_json)
    print(f"\n最终结果已保存到: final_result.json")
    
    print("\n" + "="*60)
    print("完整流程演示结束")
    print("="*60)
    
    """
    前后端集成示例（FastAPI）：
    
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import JSONResponse
    import io
    
    app = FastAPI()
    api = MinerUProductionAPI(backend="http-client", server_url="...")
    
    @app.post("/api/detect_layout")
    async def detect_layout_endpoint(file: UploadFile = File(...)):
        '''版面识别接口'''
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        result = api.detect_layout(image, image_id=file.filename)
        return JSONResponse(content=result.to_dict())
    
    @app.post("/api/extract_blocks")
    async def extract_blocks_endpoint(
        file: UploadFile = File(...),
        request_data: dict  # 包含用户修改后的blocks
    ):
        '''区块内容提取接口'''
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # 解析用户修改后的区块
        extraction_request = BlockExtractionRequest.from_dict(request_data)
        
        result = api.extract_blocks_content(
            image=image,
            blocks=extraction_request.blocks,
            image_id=extraction_request.image_id
        )
        return JSONResponse(content=result.to_dict())
    """

