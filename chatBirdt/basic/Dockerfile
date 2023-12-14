FROM node:17.0.1 as build

WORKDIR /basic
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run prod

FROM nginx:1.19

COPY ./nginx/nginx.conf /ect/nginx/nginx.conf
COPY --from=build /basic/dist/angular-tour-of-heroes/ /usr/share/nginx/html

# az login            ?
# sudo az acr login --name acrChatbirdt ?// user name and key on website

#sudo docker tag basic-angular-app acrchatbirdt.azurecr.io/basic-angular-app
#sudo docker push acrchatbirdt.azurecr.io/basic-angular-app
