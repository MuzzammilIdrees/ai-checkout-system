"""
database/seed.py — Idempotent seed script for the AI Checkout System database.

Maps all 60 MVTec D2S category names to realistic product names with plausible
retail prices (USD) and randomised initial stock quantities (seed=42).

Usage:
    python -m database.seed            # Seed (skip existing)
    python -m database.seed --reset    # Wipe and re-seed from scratch
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

# Add parent directory to path so we can import database modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from sqlalchemy.orm import Session

from database.db import (
    AdminUser,
    Base,
    Product,
    create_all_tables,
    get_engine,
    get_session_factory,
)
from utils.security import generate_salt, hash_password

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ────────────────────────── D2S Product Catalog ──────────────────────────
# Maps each MVTec D2S category name to (display_name, category_group, price_usd)
# Prices are realistic retail USD values.

D2S_PRODUCTS: list[dict] = [
    # === Beverages (IDs 1-11) ===
    {"d2s_name": "adelholzener_alpenquelle_classic_075",    "name": "Adelholzener Classic Water 0.75L",             "category": "Beverages",       "price": 1.29},
    {"d2s_name": "adelholzener_alpenquelle_naturell_075",   "name": "Adelholzener Naturell Water 0.75L",            "category": "Beverages",       "price": 1.29},
    {"d2s_name": "adelholzener_classic_bio_apfelschorle_02","name": "Adelholzener Bio Apple Spritzer 0.2L",         "category": "Beverages",       "price": 0.99},
    {"d2s_name": "adelholzener_classic_naturell_02",        "name": "Adelholzener Classic Naturell 0.2L",           "category": "Beverages",       "price": 0.89},
    {"d2s_name": "adelholzener_gourmet_mineralwasser_02",   "name": "Adelholzener Gourmet Mineral Water 0.2L",      "category": "Beverages",       "price": 1.19},
    {"d2s_name": "augustiner_lagerbraeu_hell_05",           "name": "Augustiner Lagerbier Hell 0.5L",               "category": "Beverages",       "price": 1.49},
    {"d2s_name": "augustiner_weissbier_05",                 "name": "Augustiner Weissbier 0.5L",                    "category": "Beverages",       "price": 1.59},
    {"d2s_name": "coca_cola_05",                            "name": "Coca-Cola Original 0.5L",                      "category": "Beverages",       "price": 1.29},
    {"d2s_name": "coca_cola_light_05",                      "name": "Coca-Cola Light 0.5L",                         "category": "Beverages",       "price": 1.29},
    {"d2s_name": "suntory_gokuri_lemonade",                 "name": "Suntory Gokuri Lemonade",                      "category": "Beverages",       "price": 2.49},
    {"d2s_name": "tegernseer_hell_03",                      "name": "Tegernseer Hell 0.33L",                        "category": "Beverages",       "price": 1.39},
    # === Cereal Bars & Snacks (IDs 12-15) ===
    {"d2s_name": "corny_nussvoll",                          "name": "Corny Nussvoll Bar Pack",                      "category": "Snacks",          "price": 2.49},
    {"d2s_name": "corny_nussvoll_single",                   "name": "Corny Nussvoll Single Bar",                    "category": "Snacks",          "price": 0.79},
    {"d2s_name": "corny_schoko_banane",                     "name": "Corny Schoko Banane Bar Pack",                 "category": "Snacks",          "price": 2.49},
    {"d2s_name": "corny_schoko_banane_single",              "name": "Corny Schoko Banane Single Bar",               "category": "Snacks",          "price": 0.79},
    # === Cereals (IDs 16-18) ===
    {"d2s_name": "dr_oetker_vitalis_knuspermuesli_klassisch","name": "Dr. Oetker Vitalis Knusper Müsli 600g",       "category": "Breakfast",       "price": 4.49},
    {"d2s_name": "koelln_muesli_fruechte",                  "name": "Kölln Früchte Müsli 500g",                    "category": "Breakfast",       "price": 3.99},
    {"d2s_name": "koelln_muesli_schoko",                    "name": "Kölln Schoko Müsli 500g",                     "category": "Breakfast",       "price": 3.99},
    # === Coffee & Cocoa (IDs 19-24) ===
    {"d2s_name": "caona_cocoa",                             "name": "Caona Cocoa Drink Powder 250g",                "category": "Beverages",       "price": 3.49},
    {"d2s_name": "cocoba_cocoa",                            "name": "Cocoba Breakfast Cocoa with Honey 250g",       "category": "Beverages",       "price": 3.99},
    {"d2s_name": "cafe_wunderbar_espresso",                 "name": "Café Wunderbar Espresso 250g",                 "category": "Beverages",       "price": 5.99},
    {"d2s_name": "douwe_egberts_professional_ground_coffee", "name": "Douwe Egberts Ground Coffee 500g",            "category": "Beverages",       "price": 7.99},
    {"d2s_name": "gepa_bio_caffe_crema",                    "name": "GEPA Bio Caffè Crema 250g",                   "category": "Beverages",       "price": 6.49},
    {"d2s_name": "gepa_italienischer_bio_espresso",         "name": "GEPA Italian Bio Espresso 250g",              "category": "Beverages",       "price": 6.99},
    # === Fruits (IDs 25-39) ===
    {"d2s_name": "apple_braeburn_bundle",                   "name": "Braeburn Apples Bundle",                       "category": "Fresh Produce",   "price": 2.99},
    {"d2s_name": "apple_golden_delicious",                  "name": "Golden Delicious Apple (each)",                "category": "Fresh Produce",   "price": 0.69},
    {"d2s_name": "apple_granny_smith",                      "name": "Granny Smith Apple (each)",                    "category": "Fresh Produce",   "price": 0.79},
    {"d2s_name": "apple_red_boskoop",                       "name": "Red Boskoop Apple (each)",                     "category": "Fresh Produce",   "price": 0.89},
    {"d2s_name": "avocado",                                 "name": "Avocado (each)",                               "category": "Fresh Produce",   "price": 1.49},
    {"d2s_name": "banana_bundle",                           "name": "Banana Bundle",                                "category": "Fresh Produce",   "price": 1.29},
    {"d2s_name": "banana_single",                           "name": "Banana Single (each)",                         "category": "Fresh Produce",   "price": 0.29},
    {"d2s_name": "clementine",                              "name": "Clementines Net 1kg",                          "category": "Fresh Produce",   "price": 2.99},
    {"d2s_name": "clementine_single",                       "name": "Clementine (each)",                            "category": "Fresh Produce",   "price": 0.39},
    {"d2s_name": "grapes_green_sugraone_seedless",          "name": "Green Sugraone Seedless Grapes 500g",          "category": "Fresh Produce",   "price": 3.49},
    {"d2s_name": "grapes_sweet_celebration_seedless",       "name": "Sweet Celebration Seedless Grapes 500g",       "category": "Fresh Produce",   "price": 3.99},
    {"d2s_name": "kiwi",                                    "name": "Kiwi (each)",                                  "category": "Fresh Produce",   "price": 0.49},
    {"d2s_name": "orange_single",                           "name": "Orange (each)",                                "category": "Fresh Produce",   "price": 0.59},
    {"d2s_name": "oranges",                                 "name": "Oranges Net 2kg",                              "category": "Fresh Produce",   "price": 3.99},
    {"d2s_name": "pear",                                    "name": "Pear (each)",                                  "category": "Fresh Produce",   "price": 0.69},
    # === Pasta (IDs 40-42) ===
    {"d2s_name": "pasta_reggia_elicoidali",                 "name": "Pasta Reggia Elicoidali 500g",                 "category": "Pantry",          "price": 1.49},
    {"d2s_name": "pasta_reggia_fusilli",                    "name": "Pasta Reggia Fusilli 500g",                    "category": "Pantry",          "price": 1.29},
    {"d2s_name": "pasta_reggia_spaghetti",                  "name": "Pasta Reggia Spaghetti 500g",                  "category": "Pantry",          "price": 1.29},
    # === Office Supplies (IDs 43-44) ===
    {"d2s_name": "franken_tafelreiniger",                   "name": "Franken Board Eraser",                         "category": "Office",          "price": 3.99},
    {"d2s_name": "pelikan_tintenpatrone_canon",             "name": "Pelikan Ink Cartridge for Canon",              "category": "Office",          "price": 12.99},
    # === Tea (IDs 45-51) ===
    {"d2s_name": "ethiquable_gruener_tee_ceylon",           "name": "Ethiquable Green Tea Ceylon 20 Bags",          "category": "Beverages",       "price": 3.29},
    {"d2s_name": "gepa_bio_und_fair_fencheltee",            "name": "GEPA Bio Fennel Tea 20 Bags",                 "category": "Beverages",       "price": 2.99},
    {"d2s_name": "gepa_bio_und_fair_kamillentee",           "name": "GEPA Bio Chamomile Tea 20 Bags",              "category": "Beverages",       "price": 2.99},
    {"d2s_name": "gepa_bio_und_fair_kraeuterteemischung",   "name": "GEPA Bio Herbal Tea Mix 20 Bags",             "category": "Beverages",       "price": 2.99},
    {"d2s_name": "gepa_bio_und_fair_pfefferminztee",        "name": "GEPA Bio Peppermint Tea 20 Bags",             "category": "Beverages",       "price": 2.99},
    {"d2s_name": "gepa_bio_und_fair_rooibostee",            "name": "GEPA Bio Rooibos Tea 20 Bags",                "category": "Beverages",       "price": 3.29},
    {"d2s_name": "kilimanjaro_tea_earl_grey",               "name": "Kilimanjaro Earl Grey Tea 20 Bags",           "category": "Beverages",       "price": 3.49},
    # === Vegetables (IDs 52-60) ===
    {"d2s_name": "cucumber",                                "name": "Cucumber (each)",                              "category": "Fresh Produce",   "price": 0.79},
    {"d2s_name": "carrot",                                  "name": "Carrots (each)",                               "category": "Fresh Produce",   "price": 0.49},
    {"d2s_name": "corn_salad",                              "name": "Corn Salad (Lamb's Lettuce) 100g",             "category": "Fresh Produce",   "price": 1.99},
    {"d2s_name": "lettuce",                                 "name": "Lettuce Head (each)",                          "category": "Fresh Produce",   "price": 1.29},
    {"d2s_name": "vine_tomatoes",                           "name": "Vine Tomatoes 500g",                           "category": "Fresh Produce",   "price": 2.49},
    {"d2s_name": "roma_vine_tomatoes",                      "name": "Roma Vine Tomatoes 500g",                      "category": "Fresh Produce",   "price": 2.79},
    {"d2s_name": "rocket",                                  "name": "Rocket (Arugula) 100g",                        "category": "Fresh Produce",   "price": 1.99},
    {"d2s_name": "salad_iceberg",                           "name": "Iceberg Salad Head (each)",                    "category": "Fresh Produce",   "price": 1.09},
    {"d2s_name": "zucchini",                                "name": "Zucchini (each)",                              "category": "Fresh Produce",   "price": 0.89},
]


def seed_database(session: Session, reset: bool = False) -> None:
    """
    Seed the products table with all 60 D2S product categories.

    This function is idempotent — running it twice will not create duplicates.
    Existing products are skipped unless --reset is used.

    Args:
        session: An active SQLAlchemy session.
        reset: If True, delete all existing products before re-seeding.
    """
    if reset:
        logger.warning("🔄 Resetting database — deleting all existing data...")
        session.query(Product).delete()
        session.commit()
        logger.info("✅ All existing products deleted.")

    # Use fixed seed for reproducible stock quantities
    rng = random.Random(42)

    added_count = 0
    skipped_count = 0

    for product_data in D2S_PRODUCTS:
        # Check if product already exists (idempotent)
        existing = session.query(Product).filter_by(name=product_data["name"]).first()
        if existing:
            skipped_count += 1
            continue

        product = Product(
            name=product_data["name"],
            category=product_data["category"],
            unit_price=product_data["price"],
            stock_quantity=rng.randint(20, 100),
            barcode=f"D2S-{product_data['d2s_name'].upper().replace('_', '-')[:20]}",
            image_url=None,
        )
        session.add(product)
        added_count += 1

    session.commit()
    logger.info(f"✅ Seed complete: {added_count} products added, {skipped_count} skipped (already existed).")
    logger.info(f"📦 Total products in database: {session.query(Product).count()}")

    # Seed default admin user
    seed_admin_user(session)


def seed_admin_user(session: Session) -> None:
    """
    Create the default admin user with a strong SHA-256 hashed password.

    Default credentials:
        Username: admin
        Password: Admin@Checkout#2026!
        Email: admin@aicheckout.local

    The password is hashed using HMAC-SHA256 with a random per-user salt.
    This function is idempotent — skips if the admin user already exists.

    Args:
        session: An active SQLAlchemy session.
    """
    existing = session.query(AdminUser).filter_by(username="admin").first()
    if existing:
        logger.info("✅ Admin user already exists — skipping.")
        return

    # Strong default password
    default_password = os.getenv("ADMIN_DEFAULT_PASSWORD", "Admin@Checkout#2026!")
    default_email = os.getenv("ADMIN_EMAIL", "admin@aicheckout.local")

    salt = generate_salt()
    password_hash = hash_password(default_password, salt)

    admin = AdminUser(
        username="admin",
        email=default_email,
        password_hash=password_hash,
        salt=salt,
        is_active=True,
        mfa_enabled=True,
        failed_attempts=0,
    )
    session.add(admin)
    session.commit()

    logger.info(
        f"✅ Default admin user created:"
        f"\n   Username: admin"
        f"\n   Password: {default_password}"
        f"\n   Email: {default_email}"
        f"\n   Password hash: SHA-256 + HMAC (salt: {salt[:8]}...)"
    )


def get_d2s_name_to_product_name_map() -> dict[str, str]:
    """
    Return a mapping from D2S category names to display product names.

    Returns:
        Dictionary mapping d2s_name → display name.
    """
    return {p["d2s_name"]: p["name"] for p in D2S_PRODUCTS}


def main() -> None:
    """Entry point for the seed script — parse args and run seeding."""
    parser = argparse.ArgumentParser(description="Seed the AI Checkout System database.")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Wipe all existing data and re-seed from scratch.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=os.getenv("DB_PATH", "database/checkout.db"),
        help="Path to the SQLite database file.",
    )
    args = parser.parse_args()

    # Ensure database directory exists
    db_dir = os.path.dirname(args.db_path)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)

    engine = get_engine(args.db_path)
    create_all_tables(engine)
    SessionFactory = get_session_factory(engine)

    session = SessionFactory()
    try:
        seed_database(session, reset=args.reset)
    except Exception as e:
        session.rollback()
        logger.error(f"❌ Seed failed: {e}")
        raise
    finally:
        session.close()


if __name__ == "__main__":
    main()
