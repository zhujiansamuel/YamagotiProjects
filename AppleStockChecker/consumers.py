from channels.generic.websocket import AsyncWebsocketConsumer
import json

def group_for_all():  # 全部
    return "stream_psta_all"

def group_for_job(job_id: str) -> str:
    return f"task_{job_id}"

def group_for_stream(shop_id: int, iphone_id: int) -> str:
    return f"stream_psta_shop_{shop_id}_iphone_{iphone_id}"

class TaskProgressConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.sub_groups = set()
        user = self.scope.get("user")
        signed = self.scope.get("ws_signed", False)  # 如果你用了签名URL中间件
        if not (getattr(user, "is_authenticated", False) or signed):
            # 也可以在这里选择加入某些只读公共组
            await self.close(code=4003)
            return
        await self.accept()
        # 可选：默认加入“全部频道”，如果你想“无任何参数就能收到全部”
        await self._join(group_for_all())

    async def disconnect(self, code):
        for g in getattr(self, "sub_groups", set()):
            try:
                await self.channel_layer.group_discard(g, self.channel_name)
            except Exception:
                pass

    async def receive(self, text_data=None, bytes_data=None):
        """
        客户端协议：
        {"action":"subscribe_all"}
        {"action":"subscribe", "shop_id":1, "iphone_id":2}
        {"action":"unsubscribe", "shop_id":1, "iphone_id":2}
        """
        try:
            msg = json.loads(text_data or "{}")
        except Exception:
            return

        act = msg.get("action")
        if act == "subscribe_all":
            await self._join(group_for_all())
            await self.send(json.dumps({"ok": True, "subscribed": "all"}))
        elif act == "subscribe":
            sid, iid = msg.get("shop_id"), msg.get("iphone_id")
            if isinstance(sid, int) and isinstance(iid, int):
                await self._join(group_for_stream(sid, iid))
                await self.send(json.dumps({"ok": True, "subscribed": [sid, iid]}))
        elif act == "unsubscribe":
            sid, iid = msg.get("shop_id"), msg.get("iphone_id")
            g = group_for_stream(sid, iid)
            if g in self.sub_groups:
                await self.channel_layer.group_discard(g, self.channel_name)
                self.sub_groups.remove(g)
                await self.send(json.dumps({"ok": True, "unsubscribed": [sid, iid]}))

    async def _join(self, group_name: str):
        if not hasattr(self, "sub_groups"):
            self.sub_groups = set()
        await self.channel_layer.group_add(group_name, self.channel_name)
        self.sub_groups.add(group_name)

    async def progress_message(self, event):
        await self.send(text_data=json.dumps(event["data"]))



