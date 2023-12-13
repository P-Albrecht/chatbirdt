#FROM node:17.0.1-bullseye-slim as builder

#RUN mkdir /project
#WORKDIR /project

#RUN npm install -g @angular/cli@13

#COPY package.json package-lock.json ./
#RUN npm ci

#COPY . .
#CMD ["ng", "serve", "--host", "0.0.0.0"]

FROM node:17.0.1-bullseye-slim as build

WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run prod

FROM nginx:1.19

COPY ./nginx/nginx.conf /ect/nginx/nginx.conf
COPY --from=build /app/dist/interface/ /usr/share/nginx/html

# az login            ?
# sudo az acr login --name acrChatbirdt ?// user name and key on website

#sudo docker tag chatbirtd_interface acrchatbirdt.azurecr.io/chatbirtd_interface
#sudo docker push acrchatbirdt.azurecr.io/chatbirtd_interface

