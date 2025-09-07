#!/usr/bin/env python3
"""
One-time Yahoo Fantasy API Authentication Setup
Run this script once to authenticate and save your token.
"""

import os
import sys
import json
import webbrowser
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 70)
print("🏈 YAHOO FANTASY API - ONE-TIME AUTHENTICATION SETUP")
print("=" * 70)
print()

# Your credentials from .env
CLIENT_ID = os.getenv("YAHOO_CLIENT_ID")
CLIENT_SECRET = os.getenv("YAHOO_CLIENT_SECRET")

if not CLIENT_ID or not CLIENT_SECRET:
    print("❌ ERROR: Yahoo credentials not found in .env file")
    print("Please make sure your .env file contains:")
    print("  YAHOO_CLIENT_ID=your_client_id")
    print("  YAHOO_CLIENT_SECRET=your_client_secret")
    sys.exit(1)

print("✅ Found Yahoo credentials")
print(f"   Client ID: {CLIENT_ID[:30]}...")
print(f"   Client Secret: {CLIENT_SECRET[:10]}...")
print()

# Method 1: Using yfpy (Recommended)
print("METHOD 1: Using yfpy Library (Recommended)")
print("-" * 40)

try:
    from yfpy import YahooFantasySportsQuery
    
    print("This will:")
    print("1. Open your browser to Yahoo login")
    print("2. You login and click 'Agree' to authorize")
    print("3. Yahoo will show a verification code")
    print("4. Come back here and paste that code")
    print()
    
    input("Press Enter to start the authentication process...")
    print()
    
    # Create token directory
    token_dir = Path(".tokens")
    token_dir.mkdir(exist_ok=True)
    
    print("🌐 Opening browser for Yahoo authorization...")
    print()
    
    # Initialize - this will trigger OAuth flow
    try:
        query = YahooFantasySportsQuery(
            league_id="",  # Empty to get all leagues
            game_code="nfl",
            game_id=449,  # 2025 NFL season
            yahoo_consumer_key=CLIENT_ID,
            yahoo_consumer_secret=CLIENT_SECRET,
            browser_callback=os.environ.get("RUNTIME_ENVIRONMENT", None) != "docker" and True,  # Opens browser automatically
            env_file_location=Path("."),  # Save token to current directory
            save_token_data_to_env_file=True  # Save for reuse
        )
        
        print()
        print("✅ Authentication successful!")
        print()
        
        # Test by getting user leagues
        print("Testing connection by fetching your leagues...")
        try:
            # Get user info to verify connection
            user_games = query.get_user_games()
            print(f"✅ Connected! Found {len(user_games) if user_games else 0} games")
            
            # Try to get leagues
            user_leagues = query.get_user_leagues_by_game_key("449")  # NFL 2025 season
            if user_leagues:
                print(f"✅ Found {len(user_leagues)} leagues:")
                for i, league in enumerate(user_leagues, 1):
                    league_name = getattr(league, 'name', 'Unknown')
                    league_id = getattr(league, 'league_id', 'Unknown')
                    print(f"   {i}. {league_name} (ID: {league_id})")
            
            # Save token for MCP server use
            token_file = Path(".yahoo_token.json")
            if hasattr(query, 'oauth') and hasattr(query.oauth, 'token_data'):
                with open(token_file, 'w') as f:
                    json.dump(query.oauth.token_data, f, indent=2)
                print(f"\n✅ Token saved to {token_file}")
                print("   The MCP server can now use this token!")
            
        except Exception as e:
            print(f"⚠️  Connection test failed: {e}")
            print("   But authentication may still be successful.")

    except Exception as e:
        print(f"\n❌ Authentication failed: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure your Yahoo app is configured correctly:")
        print("   - Go to https://developer.yahoo.com/apps/")
        print("   - Check your app has 'Fantasy Sports - Read' permission")
        print("   - Redirect URI should be: oob (for out-of-band)")
        print("2. Try deleting any .yahoo_oauth or token files and retry")
        
except ImportError:
    print("❌ yfpy not installed")
    print("Install with: pip install yfpy")
    print()
    print("Falling back to Method 2...")
    print()
    
    # Method 2: Manual OAuth flow
    print("METHOD 2: Manual OAuth Flow")
    print("-" * 40)
    print()
    
    # Build authorization URL
    auth_url = (
        "https://api.login.yahoo.com/oauth2/request_auth?"
        f"client_id={CLIENT_ID}&"
        "redirect_uri=oob&"
        "response_type=code&"
        "language=en-us"
    )
    
    print("Manual authentication steps:")
    print()
    print("1. Copy this URL and open it in your browser:")
    print()
    print(auth_url)
    print()
    print("2. Login to Yahoo and click 'Agree'")
    print("3. Yahoo will show you a verification code")
    print("4. Copy that code and save it")
    print()
    print("5. You'll need to exchange this code for tokens")
    print("   (This requires additional implementation)")
    
    # Try to open browser automatically
    try:
        webbrowser.open(auth_url)
        print("✅ Browser opened automatically")
    except:
        print("⚠️  Could not open browser automatically")
        print("   Please copy the URL above and open it manually")

print()
print("=" * 70)
print("NEXT STEPS")
print("=" * 70)
print()
print("Once authenticated:")
print("1. The token is saved to .yahoo_token.json")
print("2. The MCP server will use this token automatically")
print("3. Token will auto-refresh as needed")
print()
print("To use with MCP:")
print("1. Add to your MCP config (Claude, etc.)")
print("2. The server will use the saved token")
print("3. Start making Fantasy Football API calls!")
print()
print("Need help? Check YAHOO_AUTH_REALITY.md for more details.")