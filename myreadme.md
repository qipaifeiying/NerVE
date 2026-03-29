# 运行
在docker中执行blender渲染
`xvfb-run -a blender -b --python utils/visualization_blender/batch_render_curve.py`
# 数据准备
demo下修改data_list.txt
与文件夹名称列表一致
文件夹下文件名为pc_obj.ply
点云内容为只含有xyz值的二进制文件
可以使用scripts中的convert_ply.py对ply文件进行预处理