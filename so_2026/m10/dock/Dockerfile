FROM caddy:2-alpine

RUN mkdir -p /usr/share/caddy
COPY hello.html /usr/share/caddy/

COPY Caddyfile /etc/caddy/Caddyfile
RUN addgroup --system caddyuser && adduser --system --ingroup caddyuser caddyuser
RUN setcap 'cap_net_bind_service=+ep' /usr/bin/caddy
RUN chown -R caddyuser:caddyuser /usr/share/caddy /etc/caddy

WORKDIR /usr/share/caddy

USER caddyuser

EXPOSE 80 443 8080

CMD ["caddy", "run"]
