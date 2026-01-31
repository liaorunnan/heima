from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.product import Product
from app.schemas.product import ProductCreate, ProductUpdate


def create_product(db: Session, merchant_id: int, product_create: ProductCreate) -> Product:
    """创建商品"""
    # 创建新商品
    db_product = Product(
        name=product_create.name,
        description=product_create.description,
        price=product_create.price,
        merchant_id=merchant_id
    )
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product


def get_product_by_id(db: Session, product_id: int) -> Optional[Product]:
    """根据ID获取商品"""
    return db.query(Product).filter(Product.id == product_id).first()


def get_products_by_merchant_id(db: Session, merchant_id: int) -> List[Product]:
    """根据商家ID获取商品列表"""
    return db.query(Product).filter(Product.merchant_id == merchant_id).all()


def get_all_products(db: Session) -> List[Product]:
    """获取所有商品"""
    return db.query(Product).all()


def update_product(db: Session, product_id: int, merchant_id: int, product_update: ProductUpdate) -> Optional[Product]:
    """更新商品"""
    # 查找商品
    db_product = db.query(Product).filter(
        Product.id == product_id,
        Product.merchant_id == merchant_id
    ).first()
    if not db_product:
        return None

    # 更新商品信息
    update_data = product_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(db_product, field, value)

    db.commit()
    db.refresh(db_product)
    return db_product


def delete_product(db: Session, product_id: int, merchant_id: int) -> bool:
    """删除商品"""
    # 查找商品
    db_product = db.query(Product).filter(
        Product.id == product_id,
        Product.merchant_id == merchant_id
    ).first()
    if not db_product:
        return False

    # 删除商品
    db.delete(db_product)
    db.commit()
    return True