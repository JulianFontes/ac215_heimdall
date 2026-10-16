# Heimdall frontend: Vite build served as static files by nginx.
# Data (points.csv, voxels.geojson, meta.json) is NOT baked in; bind-mount it at /usr/share/nginx/html/data.

#* Build stage
FROM node:22-alpine AS build
WORKDIR /app

# Install deps first so this layer is cached until package files change (.npmrc carries legacy-peer-deps)
COPY package.json package-lock.json .npmrc ./
RUN npm ci --no-audit --no-fund

COPY . .

# Optional Mapbox token. Vite inlines VITE_* vars into the bundle at build time, so it must be a build arg.
ARG VITE_MAPBOX_TOKEN=""
ENV VITE_MAPBOX_TOKEN=$VITE_MAPBOX_TOKEN
# Kepler.gl is a large bundle; give the build more heap than Node's default
ENV NODE_OPTIONS=--max-old-space-size=4096
RUN npm run build

#* Runtime stage
# Use a small nginx image to serve the static files
FROM nginx:1.27-alpine
COPY --from=build /app/dist /usr/share/nginx/html

# Starts the webserver that services the static files so it automatically
# brings the frontend app up when the container is started
CMD ["nginx", "-g", "daemon off;"]
