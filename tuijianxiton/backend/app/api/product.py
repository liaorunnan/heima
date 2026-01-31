from typing import List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.product import Product, ProductCreate, ProductUpdate
from app.services.product_service import (
    create_product, get_product_by_id, get_products_by_merchant_id,
    get_all_products, update_product, delete_product
)
from app.api.auth import get_current_user
from app.models.user import User

router = APIRouter(prefix="/api/products", tags=["商品"])


@router.post("", response_model=Product)
def create_new_product(product_create: ProductCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """创建商品"""
    # 验证用户是否为商家
    if current_user.role != "merchant":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有商家可以创建商品"
        )

    try:
        product = create_product(
            db=db,
            merchant_id=current_user.id,
            product_create=product_create
        )
        return product
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.get("", response_model=List[Product])
def get_products(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """获取商品列表"""
    if current_user.role == "merchant":
        # 商家只能看到自己的商品
        products = get_products_by_merchant_id(db=db, merchant_id=current_user.id)
    else:
        # 客户可以看到所有商品
        products = get_all_products(db=db)
    return products


@router.get("/{product_id}", response_model=Product)
def get_product(product_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """获取商品详情"""
    product = get_product_by_id(db=db, product_id=product_id)
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="商品不存在"
        )

    # 验证商家是否只能访问自己的商品
    if current_user.role == "merchant" and current_user.id != product.merchant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权限访问此商品"
        )

    return product


@router.put("/{product_id}", response_model=Product)
def update_existing_product(product_id: int, product_update: ProductUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """更新商品"""
    # 验证用户是否为商家
    if current_user.role != "merchant":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有商家可以更新商品"
        )

    product = update_product(
        db=db,
        product_id=product_id,
        merchant_id=current_user.id,
        product_update=product_update
    )
    if not product:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="商品不存在或无权限更新"
        )

    return product


@router.delete("/{product_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_existing_product(product_id: int, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    """删除商品"""
    # 验证用户是否为商家
    if current_user.role != "merchant":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有商家可以删除商品"
        )

    success = delete_product(
        db=db,
        product_id=product_id,
        merchant_id=current_user.id
    )
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="商品不存在或无权限删除"
        )

    return None