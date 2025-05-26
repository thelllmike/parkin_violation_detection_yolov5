# crud/payment_crud.py

import os
from typing import Optional, List

from sqlalchemy.orm import Session
from web3 import Web3
from web3.exceptions import ContractLogicError  # <— correct exception

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
            "inputs": [
                {"internalType": "string", "name": "_vehicleNumber", "type": "string"}
            ],
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
    # 1️⃣ Load user & card
    user: Optional[User] = db.query(User).get(user_id)
    if not user:
        raise ValueError(f"No user found with id={user_id}")

    card: Optional[Card] = db.query(Card).get(card_id)
    if not card or card.user_id != user_id:
        raise ValueError(f"Card id={card_id} not found for user {user_id}")

    if not user.private_key_encrypted:
        raise ValueError("User has no stored private key; cannot perform on-chain payment")

    # 2️⃣ Decrypt private key
    try:
        priv_key = decrypt_key(user.private_key_encrypted)
    except Exception as e:
        raise ValueError(f"Failed to decrypt user's private key: {e}")

    # 3️⃣ Prepare ETH amount & tx params
    wei_value = w3.to_wei(amount, "ether")
    if wei_value == 0:
        raise ValueError(f"Amount {amount} ETH too small; converted to 0 Wei")

    nonce    = w3.eth.get_transaction_count(user.eth_address)
    chain_id = w3.eth.chain_id

    # 4️⃣ Fetch dynamic fees
    block = w3.eth.get_block("pending")
    base_fee     = block.get("baseFeePerGas", w3.to_wei("20", "gwei"))
    priority_fee = w3.to_wei("2", "gwei")
    max_fee      = base_fee + (priority_fee * 2)

    # 5️⃣ SIMULATE calls to catch Solidity revert strings early
    try:
        contract.functions.depositBalance().call({
            "from":  user.eth_address,
            "value": wei_value,
        })
        contract.functions.payFee(vehicle_number).call({
            "from": user.eth_address,
        })
    except ContractLogicError as e:
        raise ValueError(f"Simulation reverted: {e.revert_message or str(e)}")

    # 6️⃣ Build & send depositBalance tx
    tx1 = contract.functions.depositBalance().build_transaction({
        "from":                 user.eth_address,
        "value":                wei_value,
        "nonce":                nonce,
        "chainId":              chain_id,
        "maxPriorityFeePerGas": priority_fee,
        "maxFeePerGas":         max_fee,
    })
    tx1["gas"] = w3.eth.estimate_gas(tx1)

    signed_tx1 = w3.eth.account.sign_transaction(tx1, priv_key)
    tx_hash1   = w3.eth.send_raw_transaction(signed_tx1.raw_transaction)
    receipt1   = w3.eth.wait_for_transaction_receipt(tx_hash1)
    if receipt1.status != 1:
        raise ValueError(f"depositBalance failed on-chain: {receipt1}")

    # 7️⃣ Build & send payFee tx
    nonce += 1
    tx2 = contract.functions.payFee(vehicle_number).build_transaction({
        "from":                 user.eth_address,
        "nonce":                nonce,
        "chainId":              chain_id,
        "maxPriorityFeePerGas": priority_fee,
        "maxFeePerGas":         max_fee,
    })
    tx2["gas"] = w3.eth.estimate_gas(tx2)

    signed_tx2 = w3.eth.account.sign_transaction(tx2, priv_key)
    tx_hash2   = w3.eth.send_raw_transaction(signed_tx2.raw_transaction)
    receipt2   = w3.eth.wait_for_transaction_receipt(tx_hash2)
    if receipt2.status != 1:
        raise ValueError(f"payFee failed on-chain: {receipt2}")

    # 8️⃣ Record payment in DB
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


def process_point_payment(db: Session, user_id: int, vehicle_number: str, amount: float) -> Transaction:
    lp: Optional[LoyaltyPoints] = db.query(LoyaltyPoints).get(user_id)
    if not lp or lp.points_balance < amount:
        raise ValueError("Insufficient points balance")
    if amount <= 0:
        raise ValueError("Amount must be positive")

    user: Optional[User] = db.query(User).get(user_id)
    if not user or not user.private_key_encrypted:
        raise ValueError("User missing or has no private key")

    try:
        priv_key = decrypt_key(user.private_key_encrypted)
    except Exception as e:
        raise ValueError(f"Failed to decrypt user's private key: {e}")

    nonce    = w3.eth.get_transaction_count(user.eth_address)
    chain_id = w3.eth.chain_id

    # Fetch dynamic fees
    block = w3.eth.get_block("pending")
    base_fee     = block.get("baseFeePerGas", w3.to_wei("20", "gwei"))
    priority_fee = w3.to_wei("2", "gwei")
    max_fee      = base_fee + (priority_fee * 2)

    # Simulate
    try:
        contract.functions.payFee(vehicle_number).call({"from": user.eth_address})
    except ContractLogicError as e:
        raise ValueError(f"Simulation reverted: {e.revert_message or str(e)}")

    # Build & send
    tx = contract.functions.payFee(vehicle_number).build_transaction({
        "from":                 user.eth_address,
        "nonce":                nonce,
        "chainId":              chain_id,
        "maxPriorityFeePerGas": priority_fee,
        "maxFeePerGas":         max_fee,
    })
    tx["gas"] = w3.eth.estimate_gas(tx)

    signed_tx = w3.eth.account.sign_transaction(tx, priv_key)
    tx_hash   = w3.eth.send_raw_transaction(signed_tx.raw_transaction)
    receipt   = w3.eth.wait_for_transaction_receipt(tx_hash)
    if receipt.status != 1:
        raise ValueError(f"payFee failed on-chain: {receipt}")

    # Deduct points & record
    lp.points_balance -= int(amount)
    db.commit()
    db.refresh(lp)

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