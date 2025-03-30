from telethon import TelegramClient
from telethon.sessions import StringSession


api = open('tg_api.txt', 'r').read().split('\n')


client = TelegramClient(StringSession(api[4]), int(api[0]), api[1],
                        system_version="4.16.30-vxCUSTOM", app_version='1.0.1')
client.start(*api[2:4])
# print(client.session.save())

# client = TelegramClient("monitoring", int(api[0]), api[1],
#                         system_version="4.16.30-vxCUSTOM", app_version='1.0.1')
# client.start()
# print("s")

async def main():
    channel = await client.get_entity("t.me/historyzx")
    print(channel.title, channel.id, type(channel))
    
    f = open('mess.txt', 'w', encoding='utf-8')
    # Получение последних 100 сообщений из канала
    async for message in client.iter_messages("t.me/historyzx", limit=100):
        print(message.text, file=f, end='\n---\n')
    
    f.close()


if __name__ == '__main__':
    with client:
        client.loop.run_until_complete(main())
