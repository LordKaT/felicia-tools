FROM node:lts-alpine AS build-stage

ENV NPM_CONFIG_LOFLEVEL=warn
ENV CI=true

WORKDIR /app

COPY package.json pnpm-lock.yaml ./
RUN npm install -g pnpm && pnpm i --frozen-lockfile

COPY . .
RUN pnpm build
FROM nginx:stable-alpine AS production-stage

COPY --from=build-stage /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

LABEL org.opencontainers.image.source="https://github.com/LordKaT/transcribe-ui"
