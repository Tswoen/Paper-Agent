def handlerChunk(is_thinking,chunk):
    if is_thinking is None:
        is_thinking = False  # 初始为非思考状态

    # 2. 根据 chunk 内容和思考状态判断类型
    state = "generating"  # 默认是生成正文
    # 处理模型思考的分隔符 ``
    if '<think>' in chunk:
        is_thinking = True
        if chunk.strip() == '<think>':
            return None, is_thinking  # 跳过纯分隔符chunk
        state = "thinking"
    elif '</think>' in chunk:
        is_thinking = False
        if chunk.strip() == '</think>':
            return None, is_thinking  # 跳过纯分隔符chunk
        state = "generating"
    elif is_thinking:
    # 处于思考状态时，后续chunk均为思考内容
        state = "thinking"
    return state, is_thinking