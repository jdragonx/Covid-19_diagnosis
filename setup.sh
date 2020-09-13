mkdir -p ~/.streamlit/

echo "
[general]\n
email = \"jonherr21@gmail.com\"\n
" > ~/.streamlit/credentials.toml

echo "\
[server]\n
headless = true\n
port = $PORT\n
enableCORS = false\n
enableXsrfProtection = false\n
" > ~/.streamlit/config.toml