"""生成 bench_data/fast/delegation.jsonl 与 fusion.jsonl（手写标注，脚本只负责排版）。

    python experiments/fast/make_data.py

标注规范见 experiments/fast/LABELING.md。
"""

from __future__ import annotations

import json
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "bench_data", "fast")

# ---- 无上下文的单轮：(文本, 是否委派) ---------------------------------------------------

SINGLE: dict[str, list[tuple[str, bool]]] = {
    "chitchat": [
        ("你好", False), ("你是谁", False), ("谢谢你", False), ("再见", False), ("早上好呀", False),
        ("今天心情不太好", False), ("你喜欢什么颜色", False), ("讲个笑话吧", False), ("嗯嗯好的", False),
        ("哈哈哈你真有意思", False), ("我今天加班到十点好累啊", False), ("你觉得猫好还是狗好", False),
        ("我刚查了一下午资料头都大了", False), ("算了不说这个了", False), ("我有点紧张明天要面试", False),
        ("你会唱歌吗", False), ("晚安", False), ("你叫什么名字", False), ("你能听懂方言吗", False),
        ("我刚才说到哪了", False), ("好的明白了", False), ("对对对就是这个意思", False),
        ("你说得有道理", False), ("我想养只猫你觉得怎么样", False), ("周末一般干点啥比较好", False),
        ("给我讲讲你今天都做了什么事情", False), ("那个 呃 你在吗", False), ("我们聊聊天吧", False),
        ("你觉得人工智能会取代人类吗", False), ("我失恋了", False), ("夸夸我", False),
        ("你吃饭了吗", False), ("我计划明天早点起床", False), ("hello", False), ("thank you so much", False),
        ("下雨天适合干什么", False), ("我好无聊啊", False), ("你最喜欢哪个季节", False),
    ],
    "knowledge": [
        ("中国的首都是哪里", False), ("一年有几个月", False), ("水的沸点是多少度", False),
        ("苹果用英文怎么说", False), ("地球绕太阳转一圈要多久", False), ("李白是哪个朝代的", False),
        ("光合作用是什么意思", False), ("一公斤等于多少斤", False), ("长城有多长", False),
        ("熊猫吃什么", False), ("什么是区块链 简单说说", False), ("谢谢用日语怎么说", False),
        ("一周有几天", False), ("太阳系最大的行星是哪个", False), ("感冒了应该多喝水吗", False),
        ("睡不着有什么办法", False), ("番茄是水果还是蔬菜", False), ("世界上最高的山是哪座", False),
        ("什么是机器学习", False), ("hello 是什么意思", False),
    ],
    "trivial_calc": [
        ("三加五等于几", False), ("一百的一半是多少", False), ("十乘以十是多少", False),
        ("二十减七等于多少", False), ("一打是几个", False), ("两个小时是多少分钟", False),
        ("九九乘法表里七八多少", False), ("一半的一半是多少", False),
    ],
    "calc": [
        ("帮我计算一下 23 乘以 47", True), ("一百二十八的平方根是多少", True), ("3847 乘以 296 等于多少", True),
        ("一万块钱存三年年利率百分之二点七五复利是多少", True), ("帮我算一下 15% 的小费 账单是 286 块", True),
        ("一百八十斤等于多少磅", True), ("把 98.6 华氏度换算成摄氏度", True),
        ("我每月工资一万二 五险一金扣多少", True), ("月供怎么算 贷款一百万三十年利率百分之三点五", True),
        ("2 的 32 次方是多少", True), ("算一下 1 到 100 所有质数的和", True), ("七百八十九除以十三等于多少", True),
        ("帮我算算 12 个人平摊 3560 块每人多少", True), ("从 1990 年 3 月 5 号到今天一共多少天", True),
        ("圆周率乘以半径 7.5 的平方是多少", True), ("一百美元能换多少人民币", True),
    ],
    "realtime": [
        ("帮我查一下明天北京的天气", True), ("最近有什么关于固态电池的新闻", True), ("今天上证指数怎么样", True),
        ("特斯拉现在股价多少", True), ("今晚湖人打谁", True), ("附近有什么好吃的火锅店", True),
        ("现在去机场堵不堵", True), ("最新的 iPhone 是哪一款", True), ("今天油价多少", True),
        ("上海迪士尼今天人多吗", True), ("查查这周末有什么电影上映", True), ("比特币现在什么价", True),
        ("明天会下雨吗 我要不要带伞", True), ("最近的地铁站在哪", True), ("what's the weather like in Tokyo tomorrow", True),
        ("世界杯昨天谁赢了", True),
    ],
    "personal": [
        ("我的航班几点起飞", True), ("帮我定个明天早上七点的闹钟", True), ("提醒我下午三点开会", True),
        ("给我妈发个微信说我晚点回家", True), ("我明天有什么安排", True), ("把客厅的灯关了", True),
        ("帮我看看有没有新邮件", True), ("帮我订一张周五去上海的高铁票", True), ("我上个月花了多少钱", True),
        ("帮我叫个车去公司", True), ("把这首歌加到我的收藏", True), ("帮我把周三的会改到周四", True),
        ("我的快递到哪了", True), ("帮我给张经理打个电话", True),
    ],
    "plan": [
        ("帮我规划一个三天的杭州旅行路线", True), ("比较一下 iPhone 和 Pixel 的拍照", True),
        ("给我做个一个月的减脂计划", True), ("五千块预算配一台游戏电脑怎么配", True),
        ("帮我调研一下国内做语音助手的公司", True), ("分析一下我这个创业想法有没有前景 做宠物上门喂养", True),
        ("帮我安排一下下周的学习计划 要准备考研数学", True), ("北京到成都自驾怎么走最合适", True),
        ("比较一下三家银行的房贷利率", True), ("帮我选一款适合老人用的手机", True),
        ("给我一份婚礼筹备清单", True), ("分析一下最近半年黄金的走势", True), ("新能源车买哪款比较好 预算二十万", True),
        ("帮我规划一下怎么在三个月内学会 Python", True),
    ],
    "write": [
        ("帮我总结一下这周的会议纪要", True), ("把这段话翻译成英文：我们下周见", True), ("写一封请假邮件给我老板", True),
        ("帮我写一首关于秋天的诗", True), ("给我写个生日祝福给我闺蜜 要长一点", True), ("帮我写一段 Python 代码读取 Excel", True),
        ("帮我写一份周报", True), ("把这篇文章总结成三点", True), ("写一篇八百字的作文 题目是我的妈妈", True),
        ("帮我起草一个租房合同", True), ("帮我把这份简历改得专业一点", True), ("给公司年会写个主持词", True),
        ("写个 SQL 统计每个月的订单数", True), ("把这封英文邮件翻译成中文", True),
    ],
    "clarify": [
        ("帮我订个", False), ("那个 就是 你帮我", False), ("我想问一下", False), ("帮我查一下那个", False),
        ("你帮我算算", False),
    ],
}

# ---- 带上下文：任务进度 / 取消 / 补充 / 打断续接 -----------------------------------------

WEATHER = {"id": "t1", "query": "查询明天北京的天气", "status": "running", "progress": ["正在查询天气接口"]}
CALC = {"id": "t1", "query": "计算 3847 乘以 296", "status": "running", "progress": []}
TRIP = {"id": "t1", "query": "规划一个三天的杭州旅行路线", "status": "running", "progress": ["已确定主要景点"]}
EMAIL = {"id": "t1", "query": "写一封请假邮件给老板", "status": "running", "progress": ["正在起草"]}
TICKET = {"id": "t1", "query": "订一张周五去上海的高铁票", "status": "running", "progress": ["正在查询车次"]}
NEWS = {"id": "t1", "query": "搜索最近关于固态电池的新闻", "status": "running", "progress": ["找到 12 篇报道"]}
PC = {"id": "t2", "query": "五千元预算配一台游戏电脑", "status": "running", "progress": []}


def h(*turns: str | tuple[str, str] | tuple[str, str, bool]) -> list[dict]:
    """h("user text", ("assistant", "text"), ("assistant", "text", True)) — alternate roles."""
    out = []
    for t in turns:
        if isinstance(t, str):
            out.append({"role": "user", "content": t})
        else:
            m = {"role": t[0], "content": t[1]}
            if len(t) > 2 and t[2]:
                m["interrupted"] = True
            out.append(m)
    return out


ACK = ("assistant", "好的，我让后台去处理，你可以先聊点别的。")

CONTEXT: list[dict] = [
    # 进度询问：不新建任务，不瞎猜答案
    dict(cat="status", text="天气查到了吗", tasks=[WEATHER], history=h("帮我查一下明天北京的天气", ACK), action="status", forbid=["晴", "雨", "度"]),
    dict(cat="status", text="好了吗", tasks=[CALC], history=h("帮我算一下 3847 乘以 296", ACK), action="status", forbid=["1138712", "113 万"]),
    dict(cat="status", text="那个结果是多少啊", tasks=[CALC], history=h("帮我算一下 3847 乘以 296", ACK), action="status", forbid=["1138712"]),
    dict(cat="status", text="还要多久", tasks=[TRIP], history=h("帮我规划一个三天的杭州旅行路线", ACK), action="status"),
    dict(cat="status", text="进度怎么样了", tasks=[EMAIL], history=h("写一封请假邮件给我老板", ACK), action="status"),
    dict(cat="status", text="票订上了没", tasks=[TICKET], history=h("帮我订一张周五去上海的高铁票", ACK), action="status", forbid=["订好了", "已经订"]),
    dict(cat="status", text="新闻都说了些啥", tasks=[NEWS], history=h("最近有什么关于固态电池的新闻", ACK), action="status"),
    dict(cat="status", text="明天到底下不下雨你先说个大概", tasks=[WEATHER], history=h("帮我查一下明天北京的天气", ACK), action="status", forbid=["会下雨", "不会下雨", "晴"]),
    dict(cat="status", text="你那边查得怎么样", tasks=[WEATHER, PC], history=h("帮我查一下明天北京的天气", ACK, "五千块预算配一台游戏电脑怎么配", ACK), action="status"),
    dict(cat="status", text="电脑那个配好了吗", tasks=[WEATHER, PC], history=h("帮我查一下明天北京的天气", ACK, "五千块预算配一台游戏电脑怎么配", ACK), action="status", forbid=["4060", "i5"]),
    dict(cat="status", text="怎么还没好", tasks=[TRIP], history=h("帮我规划一个三天的杭州旅行路线", ACK), action="status"),
    dict(cat="status", text="你先大概说说杭州第一天去哪", tasks=[TRIP], history=h("帮我规划一个三天的杭州旅行路线", ACK), action="status"),
    # 取消
    dict(cat="cancel", text="算了不用查了", tasks=[WEATHER], history=h("帮我查一下明天北京的天气", ACK), action="cancel"),
    dict(cat="cancel", text="不用算了", tasks=[CALC], history=h("帮我算一下 3847 乘以 296", ACK), action="cancel"),
    dict(cat="cancel", text="那个旅行的先别弄了", tasks=[TRIP], history=h("帮我规划一个三天的杭州旅行路线", ACK), action="cancel"),
    dict(cat="cancel", text="邮件不用写了我自己写吧", tasks=[EMAIL], history=h("写一封请假邮件给我老板", ACK), action="cancel"),
    dict(cat="cancel", text="取消吧", tasks=[TICKET], history=h("帮我订一张周五去上海的高铁票", ACK), action="cancel"),
    dict(cat="cancel", text="票先不订了 行程有变", tasks=[TICKET], history=h("帮我订一张周五去上海的高铁票", ACK), action="cancel"),
    dict(cat="cancel", text="电脑那个不要了", tasks=[WEATHER, PC], history=h("帮我查一下明天北京的天气", ACK, "五千块预算配一台游戏电脑怎么配", ACK), action="cancel"),
    dict(cat="cancel", text="停停停 别查了", tasks=[NEWS], history=h("最近有什么关于固态电池的新闻", ACK), action="cancel"),
    dict(cat="cancel", text="never mind 不需要了", tasks=[WEATHER], history=h("帮我查一下明天北京的天气", ACK), action="cancel"),
    dict(cat="cancel", text="我不想知道了", tasks=[NEWS], history=h("最近有什么关于固态电池的新闻", ACK), action="cancel"),
    # 补充条件
    dict(cat="amend", text="对了 要靠近西湖的酒店", tasks=[TRIP], history=h("帮我规划一个三天的杭州旅行路线", ACK), action="amend"),
    dict(cat="amend", text="邮件语气客气一点 就说家里有事", tasks=[EMAIL], history=h("写一封请假邮件给我老板", ACK), action="amend"),
    dict(cat="amend", text="要二等座 下午出发的", tasks=[TICKET], history=h("帮我订一张周五去上海的高铁票", ACK), action="amend"),
    dict(cat="amend", text="还有 我带着老人 别安排太累", tasks=[TRIP], history=h("帮我规划一个三天的杭州旅行路线", ACK), action="amend"),
    dict(cat="amend", text="顺便看看后天的", tasks=[WEATHER], history=h("帮我查一下明天北京的天气", ACK), action="amend"),
    dict(cat="amend", text="只要国内的新闻", tasks=[NEWS], history=h("最近有什么关于固态电池的新闻", ACK), action="amend"),
    dict(cat="amend", text="显卡要 N 卡", tasks=[WEATHER, PC], history=h("帮我查一下明天北京的天气", ACK, "五千块预算配一台游戏电脑怎么配", ACK), action="amend"),
    dict(cat="amend", text="哦对了请假是请周三一天", tasks=[EMAIL], history=h("写一封请假邮件给我老板", ACK), action="amend"),
    # 已委派问题的追问 / 没有运行中的任务时的“算了”
    dict(cat="chitchat", text="算了", history=h("讲个笑话吧", ("assistant", "好啊，有一天小明去……", True)), action="say"),
    dict(cat="chitchat", text="不用了谢谢", history=h("你好", ("assistant", "你好呀，有什么可以帮你？")), action="say"),
    # 打断后的续接
    dict(cat="interrupt", text="我是说北京不是南京", history=h("帮我查一下明天南京的天气", ("assistant", "好的，我让后台查一下南京", True)),
         tasks=[{"id": "t1", "query": "查询明天南京的天气", "status": "running", "progress": []}], action="amend"),
    dict(cat="interrupt", text="停一下我换个问题", history=h("给我讲讲你今天都做了什么事情", ("assistant", "今天我先是陪一个用户聊了", True)), action="say"),
    dict(cat="interrupt", text="说重点", history=h("什么是区块链", ("assistant", "区块链啊，说起来话长，最早要追溯到", True)), action="say"),
    dict(cat="interrupt", text="不是这个意思 我是问它能干嘛", history=h("什么是区块链", ("assistant", "区块链是一种分布式账本", True)), action="say"),
    dict(cat="interrupt", text="等等 你帮我把这个规划成三天的行程吧", history=h("杭州有什么好玩的", ("assistant", "杭州好玩的地方可多了，西湖、灵隐", True)), action="delegate", delegate=True),
    dict(cat="interrupt", text="你刚才说什么", history=h("今天好累", ("assistant", "辛苦啦，要不要", True)), action="say"),
    dict(cat="interrupt", text="嗯你继续说", history=h("给我讲个故事", ("assistant", "从前有一座山，山里有", True)), action="say"),
    dict(cat="interrupt", text="不对不对 是二十三乘以七十四", history=h("帮我计算一下 23 乘以 47", ("assistant", "好的，我让后台", True)),
         tasks=[{"id": "t1", "query": "计算 23 乘以 47", "status": "running", "progress": []}], action="amend"),
    dict(cat="interrupt", text="好了好了我知道了", history=h("睡不着怎么办", ("assistant", "可以试试睡前别看手机，泡个脚，还有", True)), action="say"),
    dict(cat="interrupt", text="那顺便帮我查下上海的", history=h("北京明天天气怎么样", ACK),
         tasks=[WEATHER], action="delegate", delegate=True),
    dict(cat="interrupt", text="再说一遍刚才那个结果", history=h("帮我计算一下 23 乘以 47", ACK, ("assistant", "关于你刚才问的 23 乘以 47，结果是 1081", True)),
         tasks=[{"id": "t1", "query": "计算 23 乘以 47", "status": "done", "progress": [], "result": "23 乘以 47 等于 1081"}], action="say"),
    # 有运行中任务时的无关闲聊 / 新请求
    dict(cat="chitchat", text="你好", tasks=[CALC], history=h("帮我算一下 3847 乘以 296", ACK), action="say", forbid=["1138712"]),
    dict(cat="chitchat", text="等的时候陪我聊聊天吧", tasks=[TRIP], history=h("帮我规划一个三天的杭州旅行路线", ACK), action="say"),
    dict(cat="chitchat", text="你觉得杭州好玩吗", tasks=[TRIP], history=h("帮我规划一个三天的杭州旅行路线", ACK), action="say"),
    dict(cat="realtime", text="再帮我查下上海明天的天气", tasks=[WEATHER], history=h("帮我查一下明天北京的天气", ACK), action="delegate", delegate=True),
    dict(cat="calc", text="另外再算一下 1234 乘以 5678", tasks=[CALC], history=h("帮我算一下 3847 乘以 296", ACK), action="delegate", delegate=True),
    dict(cat="personal", text="趁这会儿帮我定个半小时后的提醒", tasks=[TRIP], history=h("帮我规划一个三天的杭州旅行路线", ACK), action="delegate", delegate=True),
]


# ---- 融合 ---------------------------------------------------------------------------

FUSION: list[dict] = [
    dict(history=h("帮我查一下明天北京的天气", ACK), deliver=[dict(query="查询明天北京的天气", result="明天北京晴转多云，气温 18 到 27 度，北风二级")],
         must_mention_any=[["北京", "天气", "明天"]], must_contain_any=[["晴", "多云"], ["27", "二十七", "18", "十八"]]),
    dict(history=h("帮我计算一下 23 乘以 47", ACK), deliver=[dict(query="计算 23 乘以 47", result="23 * 47 等于 1081")],
         must_mention_any=[["23", "二十三", "47", "四十七", "乘"]], must_contain_any=[["1081", "一千零八十一"]]),
    dict(history=h("帮我算一下 3847 乘以 296", ("assistant", "大概一百万左右吧，我让后台精确算一下。")), deliver=[dict(query="计算 3847 乘以 296", result="3847 * 296 等于 1138712")],
         must_mention_any=[["3847", "296", "乘"]], must_contain_any=[["1138712", "113 万", "一百一十三万"]], correction=True),
    dict(history=h("特斯拉现在股价多少", ("assistant", "好像是两百多美元，我让后台确认一下。")), deliver=[dict(query="查询特斯拉当前股价", result="特斯拉（TSLA）最新股价 412.5 美元，今日上涨 2.3%")],
         must_mention_any=[["特斯拉", "股价", "TSLA"]], must_contain_any=[["412", "四百一十二"]], correction=True),
    dict(history=h("我的航班几点起飞", ACK, "好的", ("assistant", "嗯，还有别的需要吗？")), deliver=[dict(query="查询用户的航班起飞时间", result="CA1234 北京首都→上海虹桥，明天 08:35 起飞，T3 航站楼")],
         must_mention_any=[["航班", "起飞", "飞机"]], must_contain_any=[["8", "八点"], ["T3", "三号航站楼", "航站楼"]]),
    dict(history=h("帮我规划一个三天的杭州旅行路线", ACK), deliver=[dict(query="规划一个三天的杭州旅行路线", result="第一天：西湖断桥、白堤、苏堤、雷峰塔；第二天：灵隐寺、飞来峰、龙井村喝茶；第三天：西溪湿地上午，下午河坊街和南宋御街。住宿建议西湖边。")],
         must_mention_any=[["杭州", "行程", "旅行", "三天"]], must_contain_any=[["西湖"], ["灵隐", "西溪", "龙井"]], max_chars=120),
    dict(history=h("帮我查一下明天北京的天气", ACK, "那上海呢", ("assistant", "上海的我也让后台一起查。")),
         deliver=[dict(query="查询明天北京的天气", result="北京明天小雨，15 到 22 度"), dict(query="查询明天上海的天气", result="上海明天多云，20 到 28 度")],
         must_mention_any=[["北京"], ["上海"]], must_contain_any=[["雨"], ["多云"]], max_chars=120),
    dict(history=h("帮我订一张周五去上海的高铁票", ACK), deliver=[dict(query="订一张周五去上海的高铁票", status="failed", result="12306 登录失效，需要用户重新扫码登录")],
         must_mention_any=[["票", "高铁", "上海"]], must_contain_any=[["登录", "扫码"], ["没", "失败", "未", "不了"]]),
    dict(history=h("最近有什么关于固态电池的新闻", ACK, "你觉得电动车会取代油车吗", ("assistant", "长期看很有可能，不过充电和成本还是关键。")),
         deliver=[dict(query="搜索最近关于固态电池的新闻", result="1. 某车企宣布 2027 年量产半固态电池车型；2. 研究团队报告硫化物电解质循环寿命突破 2000 次；3. 多家电池厂扩产。")],
         must_mention_any=[["固态电池", "电池", "新闻"]], must_contain_any=[["量产", "2027"], ["寿命", "2000", "扩产"]], max_chars=120),
    dict(history=h("一百美元能换多少人民币", ("assistant", "差不多七百块吧，我再让后台确认下实时汇率。")), deliver=[dict(query="查询美元兑人民币实时汇率，计算 100 美元", result="当前汇率 1 美元 = 7.12 人民币，100 美元约 712 元")],
         must_mention_any=[["美元", "汇率", "人民币"]], must_contain_any=[["712", "七百一十二", "7.12"]], correction=False),
    dict(history=h("写一封请假邮件给我老板", ACK), deliver=[dict(query="写一封请假邮件给老板", result="主题：请假申请\n王总您好：因家中有事，我想于本周三请假一天，工作已交接给小李，紧急事项可电话联系。望批准，谢谢！\n张三")],
         must_mention_any=[["邮件", "请假"]], must_contain_any=[["周三", "写好", "好了", "起草"]], max_chars=100),
    dict(history=h("今晚湖人打谁", ("assistant", "今晚湖人应该是打勇士，我让后台确认一下。")), deliver=[dict(query="查询今晚湖人队的比赛对手", result="今晚湖人主场对阵掘金，北京时间 10:30 开赛")],
         must_mention_any=[["湖人", "比赛"]], must_contain_any=[["掘金"]], correction=True),
    dict(history=h("比特币现在什么价", ACK, "谢谢", ("assistant", "不客气。")), deliver=[dict(query="查询比特币实时价格", status="failed", result="行情接口超时")],
         must_mention_any=[["比特币", "价格", "行情"]], must_contain_any=[["没", "失败", "超时", "未", "不了"]]),
    dict(history=h("帮我计算一下 23 乘以 47", ("assistant", "二十三乘四十七，大约一千出头，我让后台精确算。")), deliver=[dict(query="计算 23 乘以 47", result="23 * 47 等于 1081")],
         must_mention_any=[["23", "二十三", "47", "四十七", "乘"]], must_contain_any=[["1081", "一千零八十一"]], correction=False),
]


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for cat, items in SINGLE.items():
        for i, (text, dele) in enumerate(items, 1):
            rows.append({"id": f"{cat}-{i:03d}", "cat": cat, "text": text, "delegate": dele, "action": "delegate" if dele else "say"})
    seen: dict[str, int] = {}
    for item in CONTEXT:
        cat = item["cat"]
        seen[cat] = seen.get(cat, 0) + 1
        row = {"id": f"ctx-{cat}-{seen[cat]:03d}", "cat": cat, "text": item["text"], "history": item.get("history", [])}
        if item.get("tasks"):
            row["tasks"] = item["tasks"]
        row["delegate"] = item.get("delegate", False)
        row["action"] = item["action"]
        if item.get("forbid"):
            row["forbid"] = item["forbid"]
        rows.append(row)
    with open(os.path.join(OUT, "delegation.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(os.path.join(OUT, "fusion.jsonl"), "w", encoding="utf-8") as f:
        for i, item in enumerate(FUSION, 1):
            deliver = [{"id": f"t{j}", "status": "done", **d} for j, d in enumerate(item["deliver"], 1)]
            row = {"id": f"fuse-{i:03d}", "history": item["history"], "deliver": deliver,
                   "must_mention_any": item["must_mention_any"], "must_contain_any": item["must_contain_any"],
                   "correction": item.get("correction", False), "max_chars": item.get("max_chars", 80)}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    from collections import Counter

    print(len(rows), Counter(r["action"] for r in rows), Counter(r["cat"] for r in rows))
    print("fusion", len(FUSION))


if __name__ == "__main__":
    main()
