# crud/payment_crud.py

import os
from typing import Optional, List

from sqlalchemy.orm import Session
from web3 import Web3
from web3.exceptions import ContractCustomError

from model.user_model import User
from model.cards_model import Card
from model.loyalty_points_model import LoyaltyPoints
from model.transaction_model import Transaction, PaymentMethodEnum

from schemas.payment_schemas import (
    CardCreate,
    CardOut,
    TopUpIn,
    PointsOut,
    PaymentIn,
    PaymentOut,
)


from utils.security import decrypt_key

# ─── Web3 & contract setup ───────────────────────────────────────────────────────
w3 = Web3(Web3.HTTPProvider(os.getenv("API_URL")))
contract = w3.eth.contract(
    address=w3.to_checksum_address(os.getenv("CONTRACT_ADDRESS")),
    abi=[
        {
            "inputs": [],
            "name": "depositBalance",
            "outputs": [],
            "stateMutability": "payable",
            "type": "function",
        },
        {
            "inputs": [{"internalType": "string", "name": "_vehicleNumber", "type": "string"}],
            "name": "payFee",
            "outputs": [],
            "stateMutability": "nonpayable",
            "type": "function",
        },
    ],
)
# ────────────────────────────────────────────────────────────────────────────────

def process_card_payment(
    db: Session,
    user_id: int,
    vehicle_number: str,
    amount: float,
    card_id: int
) -> Transaction:
    # 1) Load user
    user: Optional[User] = db.query(User).get(user_id)
    if not user:
        raise ValueError(f"No user found with id={user_id}")

    # 2) Verify the card belongs to this user (optional)
    card: Optional[Card] = db.query(Card).get(card_id)
    if not card or card.user_id != user_id:
        raise ValueError(f"Card id={card_id} not found for user {user_id}")

    # 3) Ensure private key exists
    if not user.private_key_encrypted:
        raise ValueError("User has no stored private key; cannot perform on-chain payment")

    # 4) Decrypt private key
    try:
        priv_key = decrypt_key(user.private_key_encrypted)
    except Exception:
        raise ValueError("Failed to decrypt user’s private key")

    # 5) Convert amount to Wei
    wei_value = w3.to_wei(amount, "ether")
    if wei_value == 0:
        raise ValueError(f"Amount {amount} ETH is too small; converted to 0 Wei")

    # 6) On-chain: depositBalance
    try:
        tx1 = contract.functions.depositBalance().build_transaction({
            "from": user.eth_address,
            "value": wei_value,
            "nonce": w3.eth.get_transaction_count(user.eth_address),
        })
        signed1 = w3.eth.account.sign_transaction(tx1, priv_key)
        w3.eth.send_raw_transaction(signed1.raw_transaction)
    except ContractCustomError as revert:
        raise ValueError(f"depositBalance reverted: {revert}")

    # 7) On-chain: payFee
    try:
        tx2 = contract.functions.payFee(vehicle_number).build_transaction({
            "from": user.eth_address,
            "nonce": w3.eth.get_transaction_count(user.eth_address),
        })
        signed2 = w3.eth.account.sign_transaction(tx2, priv_key)
        w3.eth.send_raw_transaction(signed2.raw_transaction)
    except ContractCustomError as revert:
        raise ValueError(f"payFee reverted: {revert}")

    # 8) Record transaction in DB
    payment = Transaction(
        user_id=user_id,
        vehicle_number=vehicle_number,
        amount=amount,
        paid_with=PaymentMethodEnum.CARD,
        card_id=card_id
    )
    db.add(payment)
    db.commit()
    db.refresh(payment)

    return payment


def process_point_payment(
    db: Session,
    user_id: int,
    vehicle_number: str,
    amount: float
) -> Transaction:
    # 1) Load loyalty points
    lp: Optional[LoyaltyPoints] = db.query(LoyaltyPoints).get(user_id)
    if not lp or lp.points_balance < amount:
        raise ValueError("Insufficient points balance")

    # 2) Deduct points and persist
    lp.points_balance -= int(amount)
    db.commit()
    db.refresh(lp)

    # 3) Record payment off-chain
    payment = Transaction(
        user_id=user_id,
        vehicle_number=vehicle_number,
        amount=amount,
        paid_with=PaymentMethodEnum.POINTS,
        card_id=None
    )
    db.add(payment)
    db.commit()
    db.refresh(payment)

    return payment

# ─── Card CRUD ───────────────────────────────────────────────────────────────────

def create_card(
    db: Session,
    user_id: int,
    card_token: str,
    last4: str,
    exp_month: int,
    exp_year: int
) -> Card:
    """
    Persist a new Card record for the given user.
    """
    card = Card(
        user_id=user_id,
        card_token=card_token,
        last4=last4,
        exp_month=exp_month,
        exp_year=exp_year
    )
    db.add(card)
    db.commit()
    db.refresh(card)
    return card

def get_cards(db: Session, user_id: int) -> List[Card]:
    """
    Return all cards associated with `user_id`.
    """
    return db.query(Card).filter(Card.user_id == user_id).all()

# ─── Points Top-up CRUD ─────────────────────────────────────────────────────────

def topup_points(
    db: Session,
    user_id: int,
    amount: int
) -> LoyaltyPoints:
    """
    Credit `amount` points to the user's LoyaltyPoints balance,
    creating a record if none exists.
    """
    lp = db.query(LoyaltyPoints).get(user_id)
    if not lp:
        lp = LoyaltyPoints(user_id=user_id, points_balance=amount)
        db.add(lp)
    else:
        lp.points_balance += amount

    db.commit()
    db.refresh(lp)
    return lp