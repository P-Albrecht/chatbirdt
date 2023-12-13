import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';

import { ChatComponent } from './chat/chat.component';

import { SettingsComponent } from "./settings/settings.component";

const routes: Routes = [
  { path: 'chat', component: ChatComponent },
  { path: 'settings', component: SettingsComponent },

];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})

export class AppRoutingModule { }
