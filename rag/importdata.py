from match_keyword import Yinyutl
from conf import settings


from get_data.lunwen_100 import Wenzhang,create_engine,create_db_and_tables
from sqlmodel import Field, Session, SQLModel, create_engine,select




engine = create_engine(settings.url(), echo=True)  
create_db_and_tables()

num = 7361

with Session(engine) as session:
    data = session.query(Wenzhang).all()
    for item in data:

        yinyutl = Yinyutl(meta={'id': num}, child=item.title, parent=item.content, source=['英语范文'])
        yinyutl.save()
        num += 1


