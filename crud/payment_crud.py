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
        {"inputs": [], "name": "depositBalance", "outputs": [], "stateMutability": "payable", "type": "function"},
        {"inputs": [{"internalType": "string", "name": "_vehicleNumber", "type": "string"}], "name": "payFee", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
    ],
)
# ────────────────────────────────────────────────────────────────────────────────
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
        {"inputs": [], "name": "depositBalance", "outputs": [], "stateMutability": "payable", "type": "function"},
        {"inputs": [{"internalType": "string", "name": "_vehicleNumber", "type": "string"}], "name": "payFee", "outputs": [], "stateMutability": "nonpayable", "type": "function"},
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
    user: Optional[User] = db.query(User).get(user_id)
    if not user:
        raise ValueError(f"No user found with id={user_id}")

    card: Optional[Card] = db.query(Card).get(card_id)
    if not card or card.user_id != user_id:
        raise ValueError(f"Card id={card_id} not found for user {user_id}")

    if not user.private_key_encrypted:
        raise ValueError("User has no stored private key; cannot perform on-chain payment")

    try:
        priv_key = decrypt_key(user.private_key_encrypted)
    except Exception as e:
        raise ValueError(f"Failed to decrypt user's private key: {e}")

    wei_value = w3.to_wei(amount, "ether")
    if wei_value == 0:
        raise ValueError(f"Amount {amount} ETH too small; converted to 0 Wei")

    balance = w3.eth.get_balance(user.eth_address)
    max_fee = w3.to_wei('1', 'gwei')
    priority_fee = w3.to_wei('1', 'gwei')
    nonce = w3.eth.get_transaction_count(user.eth_address)

    try:
        # 1️⃣ Build & send depositBalance transaction (includes ETH amount)
        tx1 = contract.functions.depositBalance().build_transaction({
            "from": user.eth_address,
            "value": wei_value,  # Send the ETH
            "nonce": nonce,
        })
        estimated_gas1 = w3.eth.estimate_gas(tx1)
        tx1["gas"] = estimated_gas1
        tx1["maxFeePerGas"] = max_fee
        tx1["maxPriorityFeePerGas"] = priority_fee

        signed_tx1 = w3.eth.account.sign_transaction(tx1, priv_key)
        tx_hash1 = w3.eth.send_raw_transaction(signed_tx1.raw_transaction)  # Web3.py v6
        receipt1 = w3.eth.wait_for_transaction_receipt(tx_hash1)
        if receipt1.status != 1:
            raise ValueError(f"depositBalance failed: {receipt1}")

        print(f"[CHAIN] depositBalance → {tx_hash1.hex()}")

        # 2️⃣ Build & send payFee transaction
        nonce += 1
        tx2 = contract.functions.payFee(vehicle_number).build_transaction({
            "from": user.eth_address,
            "nonce": nonce,
        })
        estimated_gas2 = w3.eth.estimate_gas(tx2)
        tx2["gas"] = estimated_gas2
        tx2["maxFeePerGas"] = max_fee
        tx2["maxPriorityFeePerGas"] = priority_fee

        signed_tx2 = w3.eth.account.sign_transaction(tx2, priv_key)
        tx_hash2 = w3.eth.send_raw_transaction(signed_tx2.raw_transaction)  # Web3.py v6
        receipt2 = w3.eth.wait_for_transaction_receipt(tx_hash2)
        if receipt2.status != 1:
            raise ValueError(f"payFee failed: {receipt2}")

        print(f"[CHAIN] payFee → {tx_hash2.hex()}")

    except Exception as e:
        raise ValueError(f"Blockchain transaction reverted: {e}")

    # Save payment record in DB
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
# ─── Other CRUD operations (unchanged) ──────────────────────────────────────────
def process_point_payment(db: Session, user_id: int, vehicle_number: str, amount: float) -> Transaction:
    lp: Optional[LoyaltyPoints] = db.query(LoyaltyPoints).get(user_id)
    if not lp or lp.points_balance < amount:
        raise ValueError("Insufficient points balance")

    if not amount or amount <= 0:
        raise ValueError("Amount must be positive")

    user: Optional[User] = db.query(User).get(user_id)
    if not user:
        raise ValueError(f"No user found with id={user_id}")

    if not user.private_key_encrypted:
        raise ValueError("User has no stored private key; cannot perform on-chain payment")

    try:
        priv_key = decrypt_key(user.private_key_encrypted)
    except Exception as e:
        raise ValueError(f"Failed to decrypt user's private key: {e}")

    max_fee = w3.to_wei('1', 'gwei')
    priority_fee = w3.to_wei('1', 'gwei')
    nonce = w3.eth.get_transaction_count(user.eth_address)

    try:
        # ✅ Only call payFee (no depositBalance)
        tx = contract.functions.payFee(vehicle_number).build_transaction({
            "from": user.eth_address,
            "nonce": nonce,
        })
        estimated_gas = w3.eth.estimate_gas(tx)
        tx["gas"] = estimated_gas
        tx["maxFeePerGas"] = max_fee
        tx["maxPriorityFeePerGas"] = priority_fee

        signed_tx = w3.eth.account.sign_transaction(tx, priv_key)
        tx_hash = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        if receipt.status != 1:
            raise ValueError(f"payFee failed: {receipt}")

        print(f"[CHAIN] payFee (points) → {tx_hash.hex()}")

    except Exception as e:
        raise ValueError(f"Blockchain transaction reverted: {e}")

    # 2️⃣ Deduct points in DB
    lp.points_balance -= int(amount)
    db.commit()
    db.refresh(lp)

    # 3️⃣ Record payment in DB
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

def create_card(db: Session, user_id: int, card_token: str, last4: str, exp_month: int, exp_year: int) -> Card:
    card = Card(user_id=user_id, card_token=card_token, last4=last4, exp_month=exp_month, exp_year=exp_year)
    db.add(card)
    db.commit()
    db.refresh(card)
    return card

def get_cards(db: Session, user_id: int) -> List[Card]:
    return db.query(Card).filter(Card.user_id == user_id).all()

def topup_points(db: Session, user_id: int, amount: int) -> LoyaltyPoints:
    lp = db.query(LoyaltyPoints).get(user_id)
    if not lp:
        lp = LoyaltyPoints(user_id=user_id, points_balance=amount)
        db.add(lp)
    else:
        lp.points_balance += amount

    db.commit()
    db.refresh(lp)
    return lp